#pragma once

#include "SimulationData.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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
    // where input[i] = sum over all connections of (connectionWeight * connectedCell.signal[i])
    __inline__ __device__ static void processCell(Object* cell, bool initMatrices);

    __inline__ __device__ static float applyActivationFunction(ActivationFunction activationFunction, float x);

    // Block dimension (one warp)
    static constexpr int BlockDim = 32;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::calcSignal(SimulationData& data, SimulationStatistics& statistics)
{
    CUDA_CHECK(blockDim.x == BlockDim);

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

        copyChannels(cell.signal.channels, cell.futureSignal.channels);
        cell.signal.numTimesSent = cell.futureSignal.numTimesSent;
    }
}

__inline__ __device__ void NeuronProcessor::clearSignal(Object* object)
{
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->typeData.cell.signal.channels[i] = 0;
    }
    object->typeData.cell.signal.numTimesSent = 0;
}

__inline__ __device__ bool NeuronProcessor::isAutoTriggered(SimulationData& data, Object* object, uint32_t autoTriggerInterval, bool isPreview)
{
    CUDA_CHECK(object->type == ObjectType_Cell);

    auto triggerInterval = max(3, autoTriggerInterval);
    if (isPreview) {
        return *data.timestep % triggerInterval == 0;
    } else {
        return (*data.timestep + object->typeData.cell.creature->id) % triggerInterval == 0;
    }
}

__inline__ __device__ bool NeuronProcessor::isManuallyTriggered(SimulationData& data, Object* object)
{
    if (abs(object->typeData.cell.signal.channels[Channels::CellTypeActivation]) < TRIGGER_THRESHOLD) {
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

    __shared__ __align__(16) float sharedAccumulatedInput[MAX_CHANNELS];
    __shared__ int sharedMinNumTimesSent;

    // Init variables
    if (laneId < MAX_CHANNELS) {
        sharedAccumulatedInput[laneId] = 0.0f;
    }
    if (laneId == 0) {
        sharedMinNumTimesSent = (numConnections > 0) ? INT_MAX : 0;
    }
    block.sync();
    
    // Accumulate weighted inputs from all connected cells
    for (int connIdx = 0; connIdx < numConnections; ++connIdx) {
        auto const& connectedObject = object->connections[connIdx].object;
        
        if (connectedObject->type != ObjectType_Cell) {
            continue;
        }
        auto& connectedCell = connectedObject->typeData.cell;
        if (connectedCell.cellState == CellState_Constructing) {
            continue;
        }
        
        if (laneId == 0) {
            sharedMinNumTimesSent = min(sharedMinNumTimesSent, connectedCell.signal.numTimesSent);
        }
        
        if (laneId < MAX_CHANNELS) {
            sharedAccumulatedInput[laneId] += connectedCell.signal.channels[laneId] * cell.neuralNetwork->connectionWeights[connIdx];
        }
    }
    block.sync();

    // matrix-vector multiplication (16x16 weights * 16 input vector)
    // Each thread computes one output channel
    if (laneId < MAX_CHANNELS) {
        int row = laneId;
        float result = 0.0f;

        // Compute dot product: weights[row][0:15] * input[0:15]
        // Weights are stored row-major, so weights[row][col] = weights[row * MAX_CHANNELS + col]
        auto const* weightsRow = &cell.neuralNetwork->weights[row * MAX_CHANNELS];
        
        // Unroll the inner loop for better performance
        #pragma unroll
        for (int col = 0; col < MAX_CHANNELS; ++col) {
            result += weightsRow[col].getValue() * sharedAccumulatedInput[col];
        }

        // Add bias
        result += cell.neuralNetwork->biases[row];

        // Apply activation function and clamp
        result = applyActivationFunction(cell.neuralNetwork->activationFunctions[row], result);
        result = max(-2.0f, min(2.0f, result));
        
        cell.futureSignal.channels[row] = result;
    }

    if (laneId == 0) {
        cell.futureSignal.numTimesSent = sharedMinNumTimesSent;
    }
}

__inline__ __device__ float NeuronProcessor::applyActivationFunction(ActivationFunction activationFunction, float x)
{
    switch (activationFunction) {
    case ActivationFunction_Sigmoid:
        return 2.0f / (1.0f + __expf(-x)) - 1.0f;
    case ActivationFunction_BinaryStep:
        return x >= NEAR_ZERO ? 1.0f : 0.0f;
    case ActivationFunction_Identity:
        return x;
    case ActivationFunction_Abs:
        return abs(x);
    case ActivationFunction_Gaussian:
        return __expf(-2 * x * x);
    }
    return 0;
}
