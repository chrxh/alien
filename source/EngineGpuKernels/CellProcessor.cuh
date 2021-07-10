#pragma once

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "Tagger.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

class CellProcessor
{
public:
    __inline__ __device__ static void init(SimulationData& data);
    __inline__ __device__ static void collisions(SimulationData& data);
    __inline__ __device__ static void initForces(SimulationData& data);
    __inline__ __device__ static void calcForces(SimulationData& data);
    __inline__ __device__ static void calcPositions(SimulationData& data);
    __inline__ __device__ static void calcVelocities(SimulationData& data);
    __inline__ __device__ static void calcAveragedVelocities(SimulationData& data);
    __inline__ __device__ static void applyAveragedVelocities(SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellProcessor::init(SimulationData& data)
{
    {
        auto& cells = data.entities.cellPointers;
        auto const partition =
            calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);

            cell->temp1 = {0, 0};
            cell->tag = 0;
        }
    }
    {
        auto const partition = calcPartition(data.entities.cellPointers.getNumEntries(), blockIdx.x, gridDim.x);
        Cell** cellPointers = &data.entities.cellPointers.at(partition.startIndex);
        data.cellMap.set_block(partition.numElements(), cellPointers);
    }
}

__inline__ __device__ void CellProcessor::collisions(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        for (float dx = -0.5f; dx < 0.51f; dx += 1.0f) {
            for (float dy = -0.5f; dy < 0.51f; dy += 1.0f) {
                Cell* otherCell = data.cellMap.get(cell->absPos + float2{dx, dy});

                if (!otherCell || otherCell == cell) {
                    continue;
                }

/*
                if (cell->getProtectionCounter_safe() > 0 || otherCell->getProtectionCounter_safe() > 0) {
                    continue;
                }
*/

                auto posDelta = cell->absPos - otherCell->absPos;
                data.cellMap.mapDisplacementCorrection(posDelta);

                auto distance = Math::length(posDelta);
                if (distance >= cudaSimulationParameters.cellMaxDistance) {
                    continue;
                }

                auto velDelta = cell->vel - otherCell->vel;
                if (Math::dot(posDelta, velDelta) < 0) {

                    bool alreadyConnected = false;
                    for (int i = 0; i < cell->numConnections; ++i) {
                        if (cell->connections[i].cell == otherCell) {
                            alreadyConnected = true;
                            break;
                        }
                    }

                    if (!alreadyConnected) {
                        auto newVel1 =
                            cell->vel - posDelta * Math::dot(velDelta, posDelta) / Math::lengthSquared(posDelta);
                        auto newVel2 =
                            otherCell->vel + posDelta * Math::dot(velDelta, posDelta) / Math::lengthSquared(posDelta);

                        atomicAdd(&cell->temp1.x, newVel1.x);
                        atomicAdd(&cell->temp1.y, newVel1.y);
                        atomicAdd(&cell->tag, 1);
                        atomicAdd(&otherCell->temp1.x, newVel2.x);
                        atomicAdd(&otherCell->temp1.y, newVel2.y);
                        atomicAdd(&otherCell->tag, 1);

                        SystemDoubleLock lock;
                        lock.init(&cell->locked, &otherCell->locked);
                        if (lock.tryLock()) {
                            __threadfence();

                            if (cell->numConnections < cell->maxConnections
                                && otherCell->numConnections < otherCell->maxConnections) {
                                bool alreadyConnected = false;
                                for (int i = 0; i < cell->numConnections; ++i) {
                                    if (cell->connections[i].cell == otherCell) {
                                        alreadyConnected = true;
                                        break;
                                    }
                                }

                                if (!alreadyConnected) {
                                    auto index = cell->numConnections++;
                                    auto otherIndex = otherCell->numConnections++;

                                    cell->connections[index].cell = otherCell;
                                    cell->connections[index].distance = distance;
                                    otherCell->connections[otherIndex].cell = cell;
                                    otherCell->connections[otherIndex].distance = distance;
                                }
                            }
                            __threadfence();
                            lock.releaseLock();
                        }
                    }
                }
            }
        }
    }
}

__inline__ __device__ void CellProcessor::initForces(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->tag > 0) {
            cell->vel = cell->temp1 / cell->tag;
        }
        cell->temp1 = {0, 0};
        cell->tag = 0;
    }
}

__inline__ __device__ void CellProcessor::calcForces(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        float2 force{0, 0};
        float2 prevDisplacement;
        for (int index = 0; index < cell->numConnections; ++index) {
            auto connectingCell = cell->connections[index].cell;
            if (connectingCell->alive == 0) {
                continue;
            }

            auto displacement = connectingCell->absPos - cell->absPos;
            data.cellMap.mapDisplacementCorrection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = cell->connections[index].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::normalized(displacement) * deviation / 2;

/*
            if (index > 0) {
                auto angle = Math::angleOfVector(displacement);
                auto prevAngle = Math::angleOfVector(prevDisplacement);
                auto actualangleFromPrevious = Math::subtractAngle(angle, prevAngle);
                if (actualangleFromPrevious > 180) {
                    actualangleFromPrevious = abs(actualangleFromPrevious - 360.0f);
                }
                auto deviation = actualangleFromPrevious - cell->connections[index].angleFromPrevious;
                auto correctionMovementForLowAngle = Math::normalized((displacement + prevDisplacement) / 2);

                auto forceInc = correctionMovementForLowAngle * deviation / -3000;
                force = force + forceInc;
                atomicAdd(&connectingCell->temp1.x, -forceInc.x / 2);
                atomicAdd(&connectingCell->temp1.y, -forceInc.y / 2);
                atomicAdd(&cell->connections[index - 1].cell->temp1.x, -forceInc.x / 2);
                atomicAdd(&cell->connections[index - 1].cell->temp1.y, -forceInc.y / 2);
            }
*/
            prevDisplacement = displacement;
        }
        atomicAdd(&cell->temp1.x, force.x);
        atomicAdd(&cell->temp1.y, force.y);
    }
}

__inline__ __device__ void CellProcessor::calcPositions(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->absPos = cell->absPos + cell->vel + cell->temp1 / 2;
        data.cellMap.mapPosCorrection(cell->absPos);
        cell->temp2 = cell->temp1;  //forces
        cell->temp1 = {0, 0};
    }
}

__inline__ __device__ void CellProcessor::calcVelocities(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->vel = cell->vel + (cell->temp1 + cell->temp2) / 2;
    }
}

__inline__ __device__ void CellProcessor::calcAveragedVelocities(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    constexpr float preserveVelocityFactor = 0.8f;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto averagedVel = cell->vel * (1.0f - preserveVelocityFactor);
        for (int index = 0; index < cell->numConnections; ++index) {
            auto connectingCell = cell->connections[index].cell;
            if (connectingCell->alive == 0) {
                continue;
            }
            averagedVel = averagedVel + connectingCell->vel * (1.0f - preserveVelocityFactor);
        }
        cell->temp1 = cell->vel * preserveVelocityFactor + averagedVel / (cell->numConnections + 1);
    }
}

__inline__ __device__ void CellProcessor::applyAveragedVelocities(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->vel = cell->temp1;
    }
}
