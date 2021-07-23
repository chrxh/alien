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
    __inline__ __device__ void init(SimulationData& data);
    __inline__ __device__ void collisions(SimulationData& data);
    __inline__ __device__ void initForces(SimulationData& data);
    __inline__ __device__ void calcForces(SimulationData& data);
    __inline__ __device__ void calcPositions(SimulationData& data);
    __inline__ __device__ void calcVelocities(SimulationData& data);
    __inline__ __device__ void calcAveragedVelocities(SimulationData& data);
    __inline__ __device__ void applyAveragedVelocities(SimulationData& data);
    __inline__ __device__ void radiation(SimulationData& data);
    __inline__ __device__ void processAddConnectionOperations(SimulationData& data);
    __inline__ __device__ void processDelCellOperations(SimulationData& data);

private:
    __inline__ __device__ void scheduleCellForDestruction(Cell* cell1, int cellIndex);

    __inline__ __device__ void connectCells(Cell* cell1, Cell* cell2);
    __inline__ __device__ void destroyCell(Cell* cell, int cellIndex);

    __inline__ __device__ void addConnection(Cell* cell1, Cell* cell2, float2 const& posDelta);
    __inline__ __device__ void delConnection(Cell* cell1, Cell* cell2);

    SimulationData* _data;
    PartitionData _partition;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellProcessor::init(SimulationData& data)
{
    {
        auto& cells = data.entities.cellPointers;
        _partition =
            calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
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
    _data = &data;
    auto& cells = data.entities.cellPointers;
    _partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
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
                if (distance >= cudaSimulationParameters.cellMaxDistance
                    /*|| distance <= cudaSimulationParameters.cellMinDistance*/) {
                    continue;
                }

                if (distance < cudaSimulationParameters.cellMinDistance && cell->numConnections > 1) {
                    scheduleCellForDestruction(cell, index);
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

                        if (cell->numConnections < cell->maxConnections
                            && otherCell->numConnections < otherCell->maxConnections
                            && Math::length(velDelta) >= cudaSimulationParameters.cellFusionVelocity) {
                            auto index = atomicAdd(data.numAddConnectionOperations, 1);
                            auto& operation = data.addConnectionOperations[index];
                            operation.cell = cell;
                            operation.otherCell = otherCell;
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
    _data = &data;

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->tag > 0) {
            auto newVel = cell->temp1 / cell->tag;
            if (Math::length(newVel - cell->vel) > cudaSimulationParameters.cellMaxForce) {
                scheduleCellForDestruction(cell, index);
            }
            cell->vel = newVel;
        }
        cell->temp1 = {0, 0};
        cell->tag = 0;
    }
}

__inline__ __device__ void CellProcessor::calcForces(SimulationData& data)
{
    _data = &data;
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        float2 force{0, 0};
        float2 prevDisplacement;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;

            auto displacement = connectingCell->absPos - cell->absPos;
            data.cellMap.mapDisplacementCorrection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = cell->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::normalized(displacement) * deviation / 2;

            if (i > 0) {
                auto angle = Math::angleOfVector(displacement);
                auto prevAngle = Math::angleOfVector(prevDisplacement);
                auto actualAngleFromPrevious = Math::subtractAngle(angle, prevAngle);
                if (actualAngleFromPrevious > 180) {
                    actualAngleFromPrevious = abs(actualAngleFromPrevious - 360.0f);
                }
                auto referenceAngleFromPrevious = cell->connections[i].angleFromPrevious;
                if (referenceAngleFromPrevious > 180) {
                    referenceAngleFromPrevious = abs(referenceAngleFromPrevious - 360.0f);
                }
                auto deviation = actualAngleFromPrevious - referenceAngleFromPrevious;
                auto correctionMovementForLowAngle = Math::normalized((displacement + prevDisplacement) / 2);

                auto forceInc = correctionMovementForLowAngle * deviation / -3000;
                force = force + forceInc;
                atomicAdd(&connectingCell->temp1.x, -forceInc.x / 2);
                atomicAdd(&connectingCell->temp1.y, -forceInc.y / 2);
                atomicAdd(&cell->connections[i - 1].cell->temp1.x, -forceInc.x / 2);
                atomicAdd(&cell->connections[i - 1].cell->temp1.y, -forceInc.y / 2);
            }
            prevDisplacement = displacement;
        }
        atomicAdd(&cell->temp1.x, force.x);
        atomicAdd(&cell->temp1.y, force.y);
    }
}

__inline__ __device__ void CellProcessor::calcPositions(SimulationData& data)
{
    _data = &data;
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->absPos = cell->absPos + cell->vel + cell->temp1 / 2;
        data.cellMap.mapPosCorrection(cell->absPos);
        cell->temp2 = cell->temp1;  //forces
        cell->temp1 = {0, 0};

        bool scheduleForDestruction = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;

            auto displacement = connectingCell->absPos - cell->absPos;
            data.cellMap.mapDisplacementCorrection(displacement);
            auto actualDistance = Math::length(displacement);
            if (actualDistance > cudaSimulationParameters.cellMaxDistance*2) {
                scheduleForDestruction = true;
            }
        }
        if (scheduleForDestruction) {
            scheduleCellForDestruction(cell, index);
        }
    }
}

__inline__ __device__ void CellProcessor::calcVelocities(SimulationData& data)
{
    _data = &data;
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto acceleration = (cell->temp1 + cell->temp2) / 2;
        cell->vel = cell->vel + acceleration;
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

__inline__ __device__ void CellProcessor::radiation(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (data.numberGen.random() < cudaSimulationParameters.radiationProb) {

            auto const cellEnergy = cell->energy;
            auto& pos = cell->absPos;
            float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
                + float2{
                    (data.numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                    (data.numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
            float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
            data.cellMap.mapPosCorrection(particlePos);

            particlePos = particlePos - particleVel;  //because particle will still be moved in current time step
            float radiationEnergy =
                powf(cellEnergy, cudaSimulationParameters.radiationExponent) * cudaSimulationParameters.radiationFactor;
            radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProb;
            radiationEnergy = 2 * radiationEnergy * data.numberGen.random();
            if (cellEnergy > 1) {
                if (radiationEnergy > cellEnergy - 1) {
                    radiationEnergy = cellEnergy - 1;
                }
                cell->energy -= radiationEnergy;

                EntityFactory _factory;
                _factory.init(&data);
                auto particle =
                    _factory.createParticle(radiationEnergy, particlePos, particleVel, {cell->metadata.color});
            }
        }
    }
}

__inline__ __device__ void CellProcessor::processAddConnectionOperations(SimulationData& data)
{
    _partition = calcPartition(*data.numAddConnectionOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    _data = &data;

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto const& operation = data.addConnectionOperations[index];
        connectCells(operation.cell, operation.otherCell);
    }
}

__inline__ __device__ void CellProcessor::processDelCellOperations(SimulationData& data)
{
    _partition =
        calcPartition(*data.numDelCellOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    _data = &data;

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto const& operation = data.delCellOperations[index];
        destroyCell(operation.cell, operation.cellIndex);
    }
}

__inline__ __device__ void CellProcessor::scheduleCellForDestruction(Cell* cell, int cellIndex)
{
    if (_data->numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
        auto index = atomicAdd(_data->numDelCellOperations, 1);
        auto& operation = _data->delCellOperations[index];
        operation.cell = cell;
        operation.cellIndex = cellIndex;
    }
}

__inline__ __device__ void CellProcessor::connectCells(Cell* cell1, Cell* cell2)
{
    SystemDoubleLock lock;
    lock.init(&cell1->locked, &cell2->locked);
    if (lock.tryLock()) {
        __threadfence();

        bool alreadyConnected = false;
        for (int i = 0; i < cell1->numConnections; ++i) {
            if (cell1->connections[i].cell == cell2) {
                alreadyConnected = true;
                break;
            }
        }

        if (!alreadyConnected) {
            auto posDelta = cell2->absPos - cell1->absPos;
            _data->cellMap.mapDisplacementCorrection(posDelta);
            addConnection(cell1, cell2, posDelta);
            addConnection(cell2, cell1, posDelta * (-1));
        }

        __threadfence();
        lock.releaseLock();
    }
}

__inline__ __device__ void CellProcessor::destroyCell(Cell* cell, int cellIndex)
{
    if (cell->tryLock()) {
        for (int i = cell->numConnections - 1; i >= 0; --i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->tryLock()) {
                __threadfence();
      
                delConnection(cell, connectedCell);
                delConnection(connectedCell, cell);

                __threadfence();
                connectedCell->releaseLock();
            }
        }
        if (0 == cell->numConnections && _data->entities.cellPointers.at(cellIndex) == cell) {
/*
            auto lastCellIndex = _data->entities.cellPointers.decNumEntriesAndReturnOrigSize() - 1;
            _data->entities.cellPointers.at(cellIndex) = _data->entities.cellPointers.at(lastCellIndex);
*/
        }
        cell->releaseLock();
    }
}

__inline__ __device__ void CellProcessor::addConnection(Cell* cell1, Cell* cell2, float2 const& posDelta)
{
    auto newAngle = Math::angleOfVector(posDelta);

    int i = 0;
    float angle = 0;
    float prevAngle = 0;
    for (; i < cell1->numConnections; ++i) {
        auto connectedCell = cell1->connections[i].cell;
        auto connectedCellDelta = connectedCell->absPos - cell1->absPos;
        _data->cellMap.mapDisplacementCorrection(connectedCellDelta);
        angle = Math::angleOfVector(connectedCellDelta);
        if (newAngle < angle) {
            break;
        }
        prevAngle = angle;
    }
    for (int j = cell1->numConnections; j > i; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }

    cell1->numConnections++;
    cell1->connections[i].cell = cell2;
    cell1->connections[i].distance = Math::length(posDelta);

    if (0 == i && 1 < cell1->numConnections) {
        cell1->connections[1].angleFromPrevious = angle - newAngle;
    }
    if (0 < i && i + 1 == cell1->numConnections) {
        cell1->connections[i].angleFromPrevious = newAngle - angle;
    }
    if (0 < i && i + 1 < cell1->numConnections) {
        auto factor = (newAngle - prevAngle) / (angle - prevAngle);
        cell1->connections[i].angleFromPrevious = cell1->connections[i].angleFromPrevious * factor;
        cell1->connections[i + 1].angleFromPrevious = cell1->connections[i].angleFromPrevious * (1 - factor);
    }
}

__inline__ __device__ void CellProcessor::delConnection(Cell* cell1, Cell* cell2)
{
    for (int i = 0; i < cell1->numConnections; ++i) {
        if (cell1->connections[i].cell == cell2) {
            if (i < cell1->numConnections - 1) {
                float angleToAdd = i > 0 ? cell1->connections[i].angleFromPrevious : 0;
                for (int j = i; j < cell1->numConnections - 1; ++j) {
                    cell1->connections[j] = cell1->connections[j + 1];
                }
                cell1->connections[i].angleFromPrevious += angleToAdd;
            }
            --cell1->numConnections;
            return;
        }
    }
}
