#pragma once

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

class CellProcessor
{
public:
    __inline__ __device__ void init(SimulationData& data);
    __inline__ __device__ void clearTag(SimulationData& data);
    __inline__ __device__ void updateMap(SimulationData& data);
    __inline__ __device__ void collisions(SimulationData& data);    //prerequisite: clearTag
    __inline__ __device__ void applyAndInitForces(SimulationData& data);    //prerequisite: tag from collisions
    __inline__ __device__ void calcForces(SimulationData& data);
    __inline__ __device__ void calcPositions(SimulationData& data);
    __inline__ __device__ void calcVelocities(SimulationData& data);
    __inline__ __device__ void calcAveragedVelocities(SimulationData& data);
    __inline__ __device__ void applyAveragedVelocities(SimulationData& data);
    __inline__ __device__ void radiation(SimulationData& data);
    __inline__ __device__ void processAddConnectionOperations(SimulationData& data);
    __inline__ __device__ void processDelOperations(SimulationData& data);
    __inline__ __device__ void decay(SimulationData& data);

private:
    __inline__ __device__ void scheduleAddConnections(Cell* cell1, Cell* cell2);
    __inline__ __device__ void scheduleDelConnections(Cell* cell, int cellIndex);
    __inline__ __device__ void scheduleDelCell(Cell* cell, int cellIndex);

    __inline__ __device__ void connectCells(Cell* cell1, Cell* cell2);
    __inline__ __device__ void delConnections(Cell* cell, int cellIndex);
    __inline__ __device__ void delCell(Cell* cell, int cellIndex);

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
    auto& cells = data.entities.cellPointers;
    _partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->temp1 = {0, 0};
    }
}

__inline__ __device__ void CellProcessor::clearTag(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    _partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->tag = 0;
    }
}

__inline__ __device__ void CellProcessor::updateMap(SimulationData& data)
{
    auto const partition = calcPartition(data.entities.cellPointers.getNumEntries(), blockIdx.x, gridDim.x);
    Cell** cellPointers = &data.entities.cellPointers.at(partition.startIndex);
    data.cellMap.set_block(partition.numElements(), cellPointers);
}

__inline__ __device__ void CellProcessor::collisions(SimulationData& data)
{
    _data = &data;
    auto& cells = data.entities.cellPointers;
    _partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    Cell* otherCells[18];
    int numOtherCells;
    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        data.cellMap.get(otherCells, numOtherCells, cell->absPos);
        for (int i = 0; i < numOtherCells; ++i) {
            Cell* otherCell = otherCells[i];

            if (!otherCell || otherCell == cell) {
                continue;
            }

            auto posDelta = cell->absPos - otherCell->absPos;
            data.cellMap.mapDisplacementCorrection(posDelta);

            auto distance = Math::length(posDelta);
            if (distance >= cudaSimulationParameters.cellMaxDistance
                /*|| distance <= cudaSimulationParameters.cellMinDistance*/) {
                continue;
            }

            if (distance < cudaSimulationParameters.cellMinDistance && cell->numConnections > 1) {
                scheduleDelConnections(cell, index);
            }

            bool alreadyConnected = false;
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (connectedCell == otherCell) {
                    alreadyConnected = true;
                    break;
                }
            }

            if (!alreadyConnected) {
                auto velDelta = cell->vel - otherCell->vel;

                auto force = Math::normalized(posDelta)
                    * (cudaSimulationParameters.cellMaxDistance - Math::length(posDelta)) / 6;
                atomicAdd(&cell->temp1.x, force.x);
                atomicAdd(&cell->temp1.y, force.y);

                auto isApproaching = Math::dot(posDelta, velDelta) < 0;
                if (cell->numConnections < cell->maxConnections && otherCell->numConnections < otherCell->maxConnections
                    && Math::length(velDelta) >= cudaSimulationParameters.cellFusionVelocity && isApproaching) {
                    scheduleAddConnections(cell, otherCell);
                }
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyAndInitForces(SimulationData& data)
{
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    _data = &data;

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto force = cell->temp1;
        if (Math::length(force) > cudaSimulationParameters.cellMaxForce) {
            scheduleDelConnections(cell, index);
        }
        cell->vel = cell->vel + force;
        cell->temp1 = {0, 0};
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

                auto forceInc = correctionMovementForLowAngle * deviation / -500;
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
            scheduleDelConnections(cell, index);
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

    auto partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cudaSimulationParameters.radiationFactor > 0
            && data.numberGen.random() < cudaSimulationParameters.radiationProb) {

            auto& pos = cell->absPos;
            float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
                + float2{
                    (data.numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                    (data.numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
            float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
            data.cellMap.mapPosCorrection(particlePos);

            auto cellEnergy = cell->energy;
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
                if (cell->energy < 0) {
                    printf("ohoh\n");
                }

                EntityFactory factory;
                factory.init(&data);
                factory.createParticle(radiationEnergy, particlePos, particleVel, {cell->metadata.color});
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

__inline__ __device__ void CellProcessor::processDelOperations(SimulationData& data)
{
    _partition =
        calcPartition(*data.numDelOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    _data = &data;

    for (int index = _partition.startIndex; index <= _partition.endIndex; ++index) {
        auto const& operation = data.delOperations[index];
        if (DelOperation::Type::DelConnections == operation.type) {
            delConnections(operation.cell, operation.cellIndex);
        }
        if (DelOperation::Type::DelCell == operation.type) {
            delCell(operation.cell, operation.cellIndex);
        }
    }
}

__inline__ __device__ void CellProcessor::decay(SimulationData& data)
{
    _data = &data;
    auto& cells = data.entities.cellPointers;
    auto partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->energy < cudaSimulationParameters.cellMinEnergy) {
            if (cell->numConnections > 0) {
                scheduleDelConnections(cell, index);
            } else {
                scheduleDelCell(cell, index);
            }
        }
    }
}

__inline__ __device__ void CellProcessor::scheduleAddConnections(Cell* cell1, Cell* cell2)
{
    auto index = atomicAdd(_data->numAddConnectionOperations, 1);
    auto& operation = _data->addConnectionOperations[index];
    operation.cell = cell1;
    operation.otherCell = cell2;
}


__inline__ __device__ void CellProcessor::scheduleDelConnections(Cell* cell, int cellIndex)
{
    if (_data->numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
        auto index = atomicAdd(_data->numDelOperations, 1);
        auto& operation = _data->delOperations[index];
        operation.type = DelOperation::Type::DelConnections;
        operation.cell = cell;
        operation.cellIndex = cellIndex;
    }
}

__inline__ __device__ void CellProcessor::scheduleDelCell(Cell* cell, int cellIndex)
{
    auto index = atomicAdd(_data->numDelOperations, 1);
    auto& operation = _data->delOperations[index];
    operation.type = DelOperation::Type::DelCell;
    operation.cell = cell;
    operation.cellIndex = cellIndex;
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

__inline__ __device__ void CellProcessor::delConnections(Cell* cell, int cellIndex)
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
        cell->releaseLock();
    }
}

__inline__ __device__ void CellProcessor::delCell(Cell* cell, int cellIndex)
{
    if (cell->tryLock()) {
        __threadfence();
        
        if (0 == cell->numConnections && cell->energy != 0/* && _data->entities.cellPointers.at(cellIndex) == cell*/) {
            EntityFactory factory;
            factory.init(_data);
            factory.createParticle(cell->energy, cell->absPos, cell->vel, {cell->metadata.color});
            cell->energy = 0;

            _data->entities.cellPointers.at(cellIndex) = nullptr;
        }

        __threadfence();
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
