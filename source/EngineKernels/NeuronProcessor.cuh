#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include <sm_60_atomic_functions.h>

namespace cg = cooperative_groups;

class NeuronProcessor
{
public:
    __inline__ __device__ static void calcSignal(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void setSignal(SimulationData& data);

    __inline__ __device__ static void clearSignal(Object* object);

    __inline__ __device__ static bool isAutoTriggered(SimulationData& data, Object* object, uint32_t autoTriggerInterval, bool isPreview = false);
    __inline__ __device__ static bool isManuallyTriggered(SimulationData& data, Object* object);
    __inline__ __device__ static bool isAutoOrManuallyTriggered(SimulationData& data, Object* cell, uint32_t autoTriggerInterval, bool isPreview = false);
    __inline__ __device__ static bool isAutoOrManuallyTriggered(SimulationData& data, Object* cell, bool autoTrigger);

private:
    // Process a single cell's neural network using classic CUDA FP32 matrix-vector multiplication
    // Neural network computation: output[row] = activation(sum(weights[row][i] * input[i]) + bias[row])
    // The input vector consists of the accumulated signals of the connected cells, the cell's memory activities and its telemetry data
    // The first input gates the memory outputs: they are only recalculated (without that input) if it exceeds the trigger threshold, otherwise they are retained
    __inline__ __device__ static void processCell(Object* cell, bool initMatrices);

    __inline__ __device__ static float calcTelemetryInput(Object* object, int telemetryIndex);
    __inline__ __device__ static float applyActivationFunction(ActivationFunction activationFunction, float x);

    // Block dimension (one warp)
    static constexpr int BlockDim = 32;

    // Input index that gates the memory outputs
    static constexpr int MemoryGateInput = 0;

    // Age at which the telemetry input saturates to 1
    static constexpr float TelemetryReferenceAge = 100000.0f;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::calcSignal(SimulationData& data, SimulationStatistics& statistics)
{
    DEVICE_CHECK(blockDim.x == BlockDim);

    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    bool firstCell = true;

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& object = objects.at(index);

        if (object->type == ObjectType_Cell && object->typeData.cell.cellState != CellState_Constructing) {
            processCell(object, firstCell);
            firstCell = false;
        }
    }
}

__inline__ __device__ void NeuronProcessor::setSignal(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto& cell = object->typeData.cell;
        if (object->typeData.cell.cellState == CellState_Constructing) {
            cell.highlightIntensity = 0;
            continue;
        }

        float signalDeviations = abs(cell.neuralActivity.signals[0] - cell.futureNeuralActivity.signals[0]);
        signalDeviations += cell.event == CellEvent_Attacked && cell.eventCounter > 0;
        cell.highlightIntensity = static_cast<uint8_t>(min(255.0f, signalDeviations * 255 / 2));

        copyChannels(cell.neuralActivity.signals, cell.futureNeuralActivity.signals);
        for (int i = 0; i < MEMORY_NEURONS_PER_CELL; ++i) {
            cell.neuralActivity.memory[i] = cell.futureNeuralActivity.memory[i];
        }
    }
}

__inline__ __device__ void NeuronProcessor::clearSignal(Object* object)
{
    for (int i = 0; i < STANDARD_NEURONS_PER_CELL; ++i) {
        object->typeData.cell.neuralActivity.signals[i] = 0;
    }
}

__inline__ __device__ bool NeuronProcessor::isAutoTriggered(SimulationData& data, Object* object, uint32_t autoTriggerInterval, bool isPreview)
{
    DEVICE_CHECK(object->type == ObjectType_Cell);

    uint64_t triggerInterval = max(static_cast<uint64_t>(TIMESTEPS_PER_CELL_FUNCTION), static_cast<uint64_t>(autoTriggerInterval));
    if (isPreview) {
        return *data.timestep % triggerInterval < TIMESTEPS_PER_CELL_FUNCTION;
    } else {
        return (*data.timestep + object->typeData.cell.creature->id) % triggerInterval < TIMESTEPS_PER_CELL_FUNCTION;
    }
}

__inline__ __device__ bool NeuronProcessor::isManuallyTriggered(SimulationData& data, Object* object)
{
    if (abs(object->typeData.cell.neuralActivity.signals[Channels::CellTypeActivation]) < TRIGGER_THRESHOLD) {
        return false;
    }
    return true;
}

__inline__ __device__ bool NeuronProcessor::isAutoOrManuallyTriggered(SimulationData& data, Object* cell, uint32_t autoTriggerInterval, bool isPreview)
{
    if (autoTriggerInterval == 0) {
        return isManuallyTriggered(data, cell);
    } else {
        if (!isAutoTriggered(data, cell, autoTriggerInterval, isPreview)) {
            return false;
        } else {
            return true;
        }
    }
}

__inline__ __device__ bool NeuronProcessor::isAutoOrManuallyTriggered(SimulationData& data, Object* cell, bool autoTrigger)
{
    if (!autoTrigger) {
        return isManuallyTriggered(data, cell);
    } else {
        return true;
    }
}

__inline__ __device__ void NeuronProcessor::processCell(Object* object, bool initMatrices)
{
    auto block = cg::this_thread_block();
    auto laneId = block.thread_rank();

    auto& cell = object->typeData.cell;
    int numConnections = object->numConnections;

    __shared__ __align__(16) float sharedInput[NEURAL_NET_INPUTS];

    // Assemble the input vector: [accumulated signals from connected cells | own memory activities | telemetry data]
    if (laneId < STANDARD_NEURONS_PER_CELL) {
        float accumulatedInput = 0.0f;
        for (int connIdx = 0; connIdx < numConnections; ++connIdx) {
            auto const& connectedObject = object->connections[connIdx].object;

            if (connectedObject->type != ObjectType_Cell) {
                continue;
            }
            auto& connectedCell = connectedObject->typeData.cell;
            if (connectedCell.cellState == CellState_Constructing) {
                continue;
            }
            accumulatedInput += connectedCell.neuralActivity.signals[laneId] * cell.neuralNetwork->connectionWeights[connIdx];
        }
        sharedInput[laneId] = accumulatedInput;
    } else if (laneId < NEURAL_NET_INPUTS) {
        if (laneId < NEURAL_NET_OUTPUTS) {
            sharedInput[laneId] = cell.neuralActivity.memory[laneId - STANDARD_NEURONS_PER_CELL];
        } else {
            sharedInput[laneId] = calcTelemetryInput(object, laneId - NEURAL_NET_OUTPUTS);
        }
    }
    block.sync();

    // Matrix-vector multiplication (12x16 weights * 16 input vector)
    // Each thread computes one output value
    if (laneId < NEURAL_NET_OUTPUTS) {
        int row = laneId;

        // The first input acts as a gate for the memory outputs: they are only recalculated if it exceeds the trigger threshold
        bool isMemoryRow = row >= STANDARD_NEURONS_PER_CELL;
        bool isMemoryGateOpen = sharedInput[MemoryGateInput] > TRIGGER_THRESHOLD;

        if (isMemoryRow && !isMemoryGateOpen) {
            cell.futureNeuralActivity.memory[row - STANDARD_NEURONS_PER_CELL] = cell.neuralActivity.memory[row - STANDARD_NEURONS_PER_CELL];
        } else {
            float result = 0.0f;

            // Weights are stored row-major, so weights[row][col] = weights[row * NEURAL_NET_INPUTS + col]
            auto const* weightsRow = &cell.neuralNetwork->weights[row * NEURAL_NET_INPUTS];

// Unroll the inner loop for better performance
#pragma unroll
            for (int col = 0; col < NEURAL_NET_INPUTS; ++col) {

                // The gate input itself does not contribute to the memory outputs
                if (isMemoryRow && col == MemoryGateInput) {
                    continue;
                }
                result += weightsRow[col].getValue() * sharedInput[col];
            }

            // Add bias
            result += cell.neuralNetwork->biases[row];

            // Apply activation function and clamp
            result = applyActivationFunction(cell.neuralNetwork->activationFunctions[row], result);
            result = max(-2.0f, min(2.0f, result));

            // The memory outputs are not visible to other cells and serve as inputs for the next execution
            if (isMemoryRow) {
                cell.futureNeuralActivity.memory[row - STANDARD_NEURONS_PER_CELL] = result;
            } else {
                cell.futureNeuralActivity.signals[row] = result;
            }
        }
    }
}

__inline__ __device__ float NeuronProcessor::calcTelemetryInput(Object* object, int telemetryIndex)
{
    auto const& cell = object->typeData.cell;
    switch (telemetryIndex) {
    case TelemetryInputs::Energy: {
        // 1 corresponds to the normal cell energy
        auto normalEnergy = max(NEAR_ZERO, cudaSimulationParameters.normalCellEnergy.value[object->color]);
        return min(2.0f, cell.usableEnergy / normalEnergy);
    }
    case TelemetryInputs::Attacked:
        return cell.event == CellEvent_Attacked && cell.eventCounter > 0 ? 1.0f : 0.0f;
    case TelemetryInputs::Age:
        return min(1.0f, toFloat(cell.age) / TelemetryReferenceAge);
    case TelemetryInputs::Speed:
        return min(2.0f, Math::length(object->vel) * 20);
    }
    return 0;
}

__inline__ __device__ float NeuronProcessor::applyActivationFunction(ActivationFunction activationFunction, float x)
{
    switch (activationFunction) {
    case ActivationFunction_Tanh:
        return tanhf(x);
    case ActivationFunction_BinaryStep:
        return x >= NEAR_ZERO ? 1.0f : 0.0f;
    case ActivationFunction_Identity:
        return x;
    case ActivationFunction_Abs:
        return abs(x);
    case ActivationFunction_Gaussian:
        return __expf(-2 * x * x);
    case ActivationFunction_Mod:
        return fmodf(fmodf(x + 1.0f, 2.0f) + 2.0f, 2.0f) - 1.0f;
    }
    return 0;
}
