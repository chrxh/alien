#pragma once

#include "EngineInterface/Enums.h"

#include "Cell.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(Token* token, SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void MuscleProcessor::process(Token* token, SimulationData& data, SimulationResult& result)
{
    auto const& sourceCell = token->sourceCell;
    auto const& cell = token->cell;
    auto& tokenMem = token->memory;
    auto command = static_cast<unsigned char>(tokenMem[Enums::Muscle_Input]) % Enums::MuscleIn_Count;

    if (Enums::MuscleIn_DoNothing == command) {
        tokenMem[Enums::Muscle_Output] = Enums::MuscleOut_Success;
        return;
    }

    auto index = getConnectionIndex(cell, sourceCell);
    auto& connection = cell->connections[index];
    auto factor =
        (Enums::MuscleIn_Contract == command || Enums::MuscleIn_ContractRelax == command) ? (1.0f / 1.2f) : 1.2f;
    auto origDistance = connection.distance;
    auto distance = origDistance * factor;

    if (sourceCell->tryLock()) {
        if (distance > cudaSimulationParameters.cellMinDistance
            && distance < cudaSimulationParameters.cellMaxCollisionDistance) {

            connection.distance = distance;

            auto connectingCell = connection.cell;
            auto otherIndex = getConnectionIndex(connectingCell, cell);
            connectingCell->connections[otherIndex].distance *= factor;
        } else {
            tokenMem[Enums::Muscle_Output] = Enums::MuscleOut_LimitReached;
            sourceCell->releaseLock();
            return;
        }

        if (Enums::MuscleIn_Contract == command || Enums::MuscleIn_Expand == command) {
            auto velInc = cell->absPos - sourceCell->absPos;
            data.cellMap.mapDisplacementCorrection(velInc);
            Math::normalize(velInc);
            cell->vel = cell->vel + velInc * (origDistance - distance) * 0.5f;
        }

        sourceCell->releaseLock();
    }

    tokenMem[Enums::Muscle_Output] = Enums::MuscleOut_Success;
    result.incMuscleActivity();
}

__inline__ __device__ int MuscleProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
{
    for (int i = 0; i < cell->numConnections; ++i) {
        if (cell->connections[i].cell == otherCell) {
            return i;
        }
    }
    return 0;
}
