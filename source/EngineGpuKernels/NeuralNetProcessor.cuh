#pragma once

#include "SimulationData.cuh"
#include "QuantityConverter.cuh"
#include "SensorProcessor.cuh"
#include "MuscleProcessor.cuh"

class NeuralNetProcessor
{
public:
    __device__ __inline__ static void scheduleOperation(Token* token, SimulationData& data);

    __device__ __inline__ static void processScheduledOperation(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processOperation(Token* token, SimulationData& data, SimulationResult& result);

    __inline__ __device__ static bool hasSensorInput(Cell* cell);
    __inline__ __device__ static bool hasMuscleAndDigestionOutput(Cell* cell);
    __inline__ __device__ static Enums::MuscleIn getAndSwapMuscleCommand(float neuralActivity, Cell* cell);


    struct SpiralLookupResult
    {
        Cell* firstMatchCell;   //nullptr = no match
        Cell* secondMatchCell;

        Cell* firstMatchPrevCell;
        Cell* secondMatchPrevCell;
    };
    __device__ __inline__ static SpiralLookupResult
    spiralLookupAlgorithm(int depth, Enums::CellFunction cellFunctionType, Cell* cell, Cell* sourceCell, SimulationData& data);

    __inline__ __device__ static float getWeight(Cell* cell, int index1, int index2);
    __inline__ __device__ static float getInput(Token* token, int index);
    __inline__ __device__ static void setOutput(Token* token, int index, float value);
    __inline__ __device__ static float sigmoid(float z);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuralNetProcessor::scheduleOperation(Token* token, SimulationData& data)
{
    data.neuralNetOperations.tryAddEntry({token});
}

__device__ __inline__ void NeuralNetProcessor::processScheduledOperation(SimulationData& data, SimulationResult& result)
{
    auto partition = calcPartition(data.neuralNetOperations.getNumEntries(), blockIdx.x, gridDim.x);
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto token = data.neuralNetOperations.at(i).token;
        processOperation(token, data, result);
    }
}

__inline__ __device__ void NeuralNetProcessor::processOperation(Token* token, SimulationData& data, SimulationResult& result)
{
    auto cell = token->cell;
    auto partition = calcPartition(8, threadIdx.x, blockDim.x);

    __shared__ float input[8];
    __shared__ float output[8];
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        input[i] = getInput(token, i);
    }
    __syncthreads();

    if (hasSensorInput(cell)) {

        __shared__ SpiralLookupResult lookupResult;
        if (0 == threadIdx.x) {
            lookupResult = spiralLookupAlgorithm(16, Enums::CellFunction_Sensor, cell, token->sourceCell, data);
        }
        __syncthreads();

        if (lookupResult.firstMatchCell) {
            char output, density, distance, angle;
            SensorProcessor::searchVicinity(
                lookupResult.firstMatchCell,
                lookupResult.firstMatchPrevCell,
                token->memory[Enums::Sensor_InMinDensity],
                token->memory[Enums::Sensor_InMaxDensity],
                token->memory[Enums::Sensor_InColor],
                output,
                density,
                distance,
                angle,
                data);
            __syncthreads();

            if (0 == threadIdx.x) {
                if (output == Enums::SensorOut_ClusterFound) {
                    input[0] = static_cast<float>(static_cast<unsigned char>(distance)) / 255.0f;
                    input[1] = static_cast<float>(angle) / 128.0f;
                } else {
                    input[0] = 1.0f;
                    input[1] = 0;
                }
            }
        }
    }
    __syncthreads();

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        float netInput = 0;
        for (int j = 0; j < 8; ++j) {
            netInput += getWeight(cell, i, j) * input[j];
        }
        output[i] = sigmoid(netInput);
        setOutput(token, i, output[i]);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        if (hasMuscleAndDigestionOutput(cell)) {
            auto lookupResult = spiralLookupAlgorithm(16, Enums::CellFunction_Muscle, cell, token->sourceCell, data);
            char discardedOutput;
            if (lookupResult.firstMatchCell) {
                MuscleProcessor::process(
                    lookupResult.firstMatchCell, lookupResult.firstMatchPrevCell, getAndSwapMuscleCommand(output[0], cell), discardedOutput, data, result);
            }
            if (lookupResult.secondMatchCell) {
                MuscleProcessor::process(
                    lookupResult.secondMatchCell, lookupResult.secondMatchPrevCell, getAndSwapMuscleCommand(output[1], cell), discardedOutput, data, result);
            }

        }
    }
}

__inline__ __device__ bool NeuralNetProcessor::hasSensorInput(Cell* cell)
{
    return (static_cast<unsigned char>(cell->staticData[0]) & 0x1) == 1;
}

__inline__ __device__ bool NeuralNetProcessor::hasMuscleAndDigestionOutput(Cell* cell)
{
    return (static_cast<unsigned char>(cell->staticData[0]) & 0x2) == 2;
}

__inline__ __device__ Enums::MuscleIn NeuralNetProcessor::getAndSwapMuscleCommand(float neuralActivity, Cell* cell)
{
    bool contraction = (static_cast<unsigned char>(cell->mutableData[0]) & 1) == 1;
    cell->mutableData[0] = cell->mutableData[0] ^ 1;    //swap bit

    //forward movement
    if (neuralActivity < 0.3) {
        if (contraction) {
            return Enums::MuscleIn_Contract;
        } else {
            return Enums::MuscleIn_ExpandRelax;
        }
    }

    //backward movement
    if (neuralActivity > 0.7) {
        if (contraction) {
            return Enums::MuscleIn_ContractRelax;
        } else {
            return Enums::MuscleIn_Expand;
        }
    }

    return Enums::MuscleIn_DoNothing;
}

__device__ __inline__ auto
NeuralNetProcessor::spiralLookupAlgorithm(int depth, Enums::CellFunction cellFunctionType, Cell* cell, Cell* sourceCell, SimulationData& data)
    -> SpiralLookupResult
{
    SpiralLookupResult result;
    result.firstMatchCell = nullptr;
    result.secondMatchCell = nullptr;

    auto visitedCellData = data.processMemory.getArray<Cell*>(256 * 2);
    HashSet<Cell*, HashFunctor<Cell*>> visitedCell(depth * 2, visitedCellData);

    auto currentCell = cell;
    auto prevCell = sourceCell;
    for (int currentDepth = 0; currentDepth < depth; ++currentDepth) {
        visitedCell.insert(currentCell);

        auto posDelta = prevCell->absPos - currentCell->absPos;
        data.cellMap.correctDirection(posDelta);
        auto originAngle = Math::angleOfVector(posDelta);

        auto nextCellFound = false;
        Cell* nextCell = nullptr;
        auto nextCellAngle = 0.0f;
        for (int i = 0; i < currentCell->numConnections; ++i) {
            auto nextCandidateCell = currentCell->connections[i].cell;
            if (!visitedCell.contains(nextCandidateCell)) {

                //calc angle from nextCandidateCell
                auto nextPosDelta = nextCandidateCell->absPos - currentCell->absPos;
                data.cellMap.correctDirection(nextPosDelta);
                auto angle = Math::angleOfVector(nextPosDelta);

                //another cell already found? => compare angles
                if (nextCellFound) {

                    //"angle" should be between "originAngle" and "nextCellAngle" in modulo arithmetic,
                    if (Math::isAngleInBetween(originAngle, nextCellAngle, angle)) {
                        nextCell = nextCandidateCell;
                        nextCellAngle = angle;
                    }

                }

                //no other cell found so far? => save cell and its angle
                else {
                    nextCell = nextCandidateCell;
                    nextCellAngle = angle;
                }
                nextCellFound = true;
            }
        }

        //next cell found?
        if (nextCellFound) {
            prevCell = currentCell;
            currentCell = nextCell;
            if (currentCell->getCellFunctionType() == cellFunctionType) {
                result.secondMatchCell = result.firstMatchCell;
                result.secondMatchPrevCell = result.firstMatchPrevCell;

                result.firstMatchCell = currentCell;
                result.firstMatchPrevCell = prevCell;
                if (result.secondMatchCell) {
                    return result;
                }
            }
        }

        //no next cell found? => finish
        else {
            return result;
        }
    }
    return result;
}

__inline__ __device__ float NeuralNetProcessor::getWeight(Cell* cell, int index1, int index2)
{
    auto flatIndex = index1 + 1 + index2 * 8;
    auto data = static_cast<unsigned char>(cell->staticData[flatIndex / 2]);
    int weigthInt = (flatIndex % 2) == 0 ? data & 0xf : (data >> 4) & 0xf;
    return static_cast<float>(weigthInt) / 16 - 0.5f;
}

__inline__ __device__ float NeuralNetProcessor::getInput(Token* token, int index)
{
    int address = Enums::NeuralNet_InOut + index * 2;
    auto byte1 = static_cast<int>(static_cast<unsigned char>(token->memory[address]));
    auto byte2 = static_cast<int>(static_cast<unsigned char>(token->memory[address + 1]));
    return static_cast<float>((byte1 << 8) + byte2) / 65536;
}

__inline__ __device__ void NeuralNetProcessor::setOutput(Token* token, int index, float value)
{
    auto scaledValue = static_cast<int>(value * 65536);
    auto byte1 = scaledValue >> 8;
    auto byte2 = scaledValue & 0xff;

    int address = Enums::NeuralNet_InOut + index * 2;
    token->memory[address] = static_cast<unsigned char>(byte1);
    token->memory[address + 1] = static_cast<unsigned char>(byte2);
}

__inline__ __device__ float NeuralNetProcessor::sigmoid(float z)
{
    return 1.0f / (1.0f + __expf(-z));
}
