#pragma once

#include "EngineInterface/ElementaryTypes.h"

#include "Cell.cuh"
#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"

class MuscleFunction
{
public:
    __inline__ __device__ static void processing(Token* token, SimulationData& data);

private:
    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void MuscleFunction::processing(Token* token, SimulationData& data)
{
    auto const& sourceCell = token->sourceCell;
    auto const& cell = token->cell;
    auto& tokenMem = token->memory;
    auto command = static_cast<unsigned char>(tokenMem[Enums::Muscle::INPUT]) % Enums::MuscleIn::_COUNTER;

    if (Enums::MuscleIn::DO_NOTHING == command) {
        tokenMem[Enums::Muscle::OUTPUT] = Enums::MuscleOut::SUCCESS;
        return;
    }

    auto index = getConnectionIndex(cell, sourceCell);
    auto& connection = cell->connections[index];
    auto factor =
        (Enums::MuscleIn::CONTRACT == command || Enums::MuscleIn::CONTRACT_RELAX == command) ? (1.0f / 1.2f) : 1.2f;
    auto origDistance = connection.distance;
    auto distance = origDistance * factor;

    if (sourceCell->tryLock()) {
        if (distance > cudaSimulationParameters.cellMinDistance
            && distance < cudaSimulationParameters.cellMaxDistance) {

            connection.distance = distance;

            auto connectingCell = connection.cell;
            auto otherIndex = getConnectionIndex(connectingCell, cell);
            connectingCell->connections[otherIndex].distance *= factor;
        } else {
            tokenMem[Enums::Muscle::OUTPUT] = Enums::MuscleOut::LIMIT_REACHED;
            sourceCell->releaseLock();
            return;
        }

        if (Enums::MuscleIn::CONTRACT == command || Enums::MuscleIn::EXPAND == command) {
            auto velInc = cell->absPos - sourceCell->absPos;
            data.cellMap.mapDisplacementCorrection(velInc);
            Math::normalize(velInc);
            cell->vel = cell->vel + velInc * (origDistance - distance) * 0.5f;
        }

        sourceCell->releaseLock();
    }

    tokenMem[Enums::Muscle::OUTPUT] = Enums::MuscleOut::SUCCESS;
}

__inline__ __device__ int MuscleFunction::getConnectionIndex(Cell* cell, Cell* otherCell)
{
    for (int i = 0; i < cell->numConnections; ++i) {
        if (cell->connections[i].cell == otherCell) {
            return i;
        }
    }
    return 0;
}
