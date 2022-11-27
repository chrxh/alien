#pragma once

#include "EngineInterface/Enums.h"

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);

    __inline__ __device__ static void contractionExpansion(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber);

    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void MuscleProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& cell = operations.at(i).cell;
        processCell(data, result, cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);

    if (cell->cellFunctionData.muscle.mode == Enums::MuscleMode_ContractionExpansion) {
        contractionExpansion(data, result, cell, inputExecutionOrderNumber);
    }

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void
MuscleProcessor::contractionExpansion(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber)
{
    //expansion
    if (cell->activity.channels[0] > cudaSimulationParameters.cellFunctionMuscleActivityThreshold) {
        if (!cell->tryLock()) {
            return;
        }
        for (int i = 0; i < cell->numConnections; ++i) {
            auto& connection = cell->connections[i];
            if (connection.cell->executionOrderNumber == inputExecutionOrderNumber) {
                if (!connection.cell->tryLock()) {
                    continue;
                }
                auto newDistance = connection.distance * cudaSimulationParameters.cellFunctionMuscleContractionExpansionFactor;
                if (newDistance >= cudaSimulationParameters.cellMaxBindingDistance * 0.8f) {
                    continue;
                }
                connection.distance = newDistance;

                auto otherIndex = getConnectionIndex(connection.cell, cell);
                connection.cell->connections[otherIndex].distance = newDistance;
                connection.cell->releaseLock();
            }
        }
        cell->releaseLock();
    }

    //contraction
    if (cell->activity.channels[0] < cudaSimulationParameters.cellFunctionMuscleOppositeActivityThreshold) {
        if (!cell->tryLock()) {
            return;
        }
        for (int i = 0; i < cell->numConnections; ++i) {
            auto& connection = cell->connections[i];
            if (connection.cell->executionOrderNumber == inputExecutionOrderNumber) {
                if (!connection.cell->tryLock()) {
                    continue;
                }
                auto newDistance = connection.distance / cudaSimulationParameters.cellFunctionMuscleContractionExpansionFactor;
                if (newDistance <= cudaSimulationParameters.cellMinDistance * 1.2f) {
                    continue;
                }
                connection.distance = newDistance;

                auto otherIndex = getConnectionIndex(connection.cell, cell);
                connection.cell->connections[otherIndex].distance = newDistance;
                connection.cell->releaseLock();
            }
        }
        cell->releaseLock();
    }
}


//__inline__ __device__ void MuscleProcessor::process(Token* token, SimulationData& data, SimulationResult& result)
//{
//    process(token->cell, token->sourceCell, token->memory[Enums::Muscle_Input], token->memory[Enums::Muscle_Output], data, result);
//}
//
//__inline__ __device__ void MuscleProcessor::process(Cell* cell, Cell* sourceCell, char input, char& output, SimulationData& data, SimulationResult& result)
//{
//    auto command = static_cast<unsigned char>(input) % Enums::MuscleIn_Count;
//
//    if (Enums::MuscleIn_DoNothing == command) {
//        output = Enums::MuscleOut_Success;
//        return;
//    }
//
//    auto index = getConnectionIndex(cell, sourceCell);
//    auto& connection = cell->connections[index];
//    auto factor = (Enums::MuscleIn_Contract == command || Enums::MuscleIn_ContractRelax == command) ? (1.0f / 1.2f) : 1.2f;
//    auto origDistance = connection.distance;
//    auto distance = origDistance * factor;
//
//    if (sourceCell->tryLock()) {
//        if (distance > cudaSimulationParameters.cellMinDistance && distance < cudaSimulationParameters.cellMaxCollisionDistance) {
//
//            connection.distance = distance;
//
//            auto connectingCell = connection.cell;
//            auto otherIndex = getConnectionIndex(connectingCell, cell);
//            connectingCell->connections[otherIndex].distance *= factor;
//        } else {
//            output = Enums::MuscleOut_LimitReached;
//            sourceCell->releaseLock();
//            return;
//        }
//
//        if (Enums::MuscleIn_Contract == command || Enums::MuscleIn_Expand == command) {
//            auto velInc = cell->absPos - sourceCell->absPos;
//            data.cellMap.correctDirection(velInc);
//            Math::normalize(velInc);
//            cell->vel = cell->vel + velInc * (origDistance - distance) * 0.5f;
//        }
//
//        sourceCell->releaseLock();
//    }
//
//    output = Enums::MuscleOut_Success;
//    result.incMuscleActivity();
//}
//
__inline__ __device__ int MuscleProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
{
    for (int i = 0; i < cell->numConnections; ++i) {
        if (cell->connections[i].cell == otherCell) {
            return i;
        }
    }
    return 0;
}
