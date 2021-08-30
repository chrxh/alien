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

    if (Enums::MuscleIn::CONTRACT == command || Enums::MuscleIn::EXPAND == command) {
        auto factor = Enums::MuscleIn::CONTRACT == command ? (1.0f / 1.2f) : 1.2f;
        for (int index = 0; index < cell->numConnections; ++index) {
            auto& connection = cell->connections[index];
            if (connection.cell->tryLock()) {

                auto distance = connection.distance * factor;
                if (distance > cudaSimulationParameters.cellMinDistance
                    && distance < cudaSimulationParameters.cellMaxDistance) {

                    connection.distance *= factor;

                    auto connectingCell = connection.cell;
                    auto otherIndex = getConnectionIndex(connectingCell, cell);
                    connectingCell->connections[otherIndex].distance *= factor;
                } else {
                    tokenMem[Enums::Muscle::OUTPUT] = Enums::MuscleOut::LIMIT_REACHED;
                    connection.cell->releaseLock();
                    return;
                }
                connection.cell->releaseLock();
            }
        }
    }
    if (Enums::MuscleIn::CONTRACT_ANGLE == command || Enums::MuscleIn::EXPAND_ANGLE == command) {
/*
        auto factor = Enums::MuscleIn::CONTRACT_ANGLE == command ? (1.0f / 1.2f) : 1.2f;
        auto index = getConnectionIndex(cell, sourceCell);
        auto& connection = cell->connections[index];
        auto origAngle = connection.angleFromPrevious;
        connection.angleFromPrevious *= factor;

        auto otherIndex = (index + 1) % cell->numConnections;
        auto& otherConnection = cell->connections[otherIndex];
        otherConnection.angleFromPrevious -= (connection.angleFromPrevious - origAngle);
*/
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
