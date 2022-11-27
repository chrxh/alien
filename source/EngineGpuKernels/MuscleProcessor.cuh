#pragma once

#include "EngineInterface/Enums.h"

#include "Cell.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
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
//__inline__ __device__ int MuscleProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
//{
//    for (int i = 0; i < cell->numConnections; ++i) {
//        if (cell->connections[i].cell == otherCell) {
//            return i;
//        }
//    }
//    return 0;
//}
