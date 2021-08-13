#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "CellConnectionProcessor.cuh"

class CellProcessor
{
public:
    __inline__ __device__ void init(SimulationData& data);
    __inline__ __device__ void clearTag(SimulationData& data);
    __inline__ __device__ void updateMap(SimulationData& data);
    __inline__ __device__ void collisions(SimulationData& data);    //prerequisite: clearTag
    __inline__ __device__ void applyAndInitForces(SimulationData& data);    //prerequisite: tag from collisions
    __inline__ __device__ void calcForces(SimulationData& data, int numCellPointers);
    __inline__ __device__ void calcPositions(SimulationData& data);
    __inline__ __device__ void calcVelocities(SimulationData& data);
    __inline__ __device__ void calcAveragedVelocities(SimulationData& data);
    __inline__ __device__ void applyAveragedVelocities(SimulationData& data);
    __inline__ __device__ void radiation(SimulationData& data);
    __inline__ __device__ void decay(SimulationData& data);

private:
    __inline__ __device__ void connectCells(Cell* cell1, Cell* cell2);
    __inline__ __device__ void delConnections(Cell* cell);
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
                CellConnectionProcessor::scheduleDelConnections(data, cell);
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
                    CellConnectionProcessor::scheduleAddConnections(data, cell, otherCell);
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
            CellConnectionProcessor::scheduleDelConnections(data, cell);
        }
        cell->vel = cell->vel + force;
        if (Math::length(cell->vel) > 2) {
            cell->vel = Math::normalized(cell->vel) * 2;
        }
        cell->temp1 = {0, 0};
    }
}

__inline__ __device__ void CellProcessor::calcForces(SimulationData& data, int numCellPointers)
{
    _data = &data;
    auto& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(numCellPointers, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (0 == cell->numConnections) {
            continue;
        }
        float2 force{0, 0};
        float2 prevDisplacement = cell->connections[cell->numConnections - 1].cell->absPos - cell->absPos;
        data.cellMap.mapDisplacementCorrection(prevDisplacement);
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;

            auto displacement = connectingCell->absPos - cell->absPos;
            data.cellMap.mapDisplacementCorrection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = cell->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::normalized(displacement) * deviation / 2;

            auto angle = Math::angleOfVector(displacement);
            auto prevAngle = Math::angleOfVector(prevDisplacement);
            auto actualAngleFromPrevious = Math::subtractAngle(angle, prevAngle);
            auto referenceAngleFromPrevious = cell->connections[i].angleFromPrevious;

            auto angleDeviation = abs(referenceAngleFromPrevious - actualAngleFromPrevious) / 2000;

            auto force1 = Math::normalized(displacement) * angleDeviation;
            Math::rotateQuarterClockwise(force1);

            auto force2 = Math::normalized(prevDisplacement) * angleDeviation;
            Math::rotateQuarterCounterClockwise(force2);

            if (referenceAngleFromPrevious < actualAngleFromPrevious) {
                force1 = force1 * (-1);
                force2 = force2 * (-1);
            }
            atomicAdd(&connectingCell->temp1.x, force1.x);
            atomicAdd(&connectingCell->temp1.y, force1.y);
            if (i > 0) {
                atomicAdd(&cell->connections[i - 1].cell->temp1.x, force2.x);
                atomicAdd(&cell->connections[i - 1].cell->temp1.y, force2.y);
            } else {
                auto lastIndex = cell->numConnections - 1;
                atomicAdd(&cell->connections[lastIndex].cell->temp1.x, force2.x);
                atomicAdd(&cell->connections[lastIndex].cell->temp1.y, force2.y);
            }
            force = force - (force1 + force2);

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
            CellConnectionProcessor::scheduleDelConnections(data, cell);
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

                EntityFactory factory;
                factory.init(&data);
                factory.createParticle(radiationEnergy, particlePos, particleVel, {cell->metadata.color});
            }
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
                CellConnectionProcessor::scheduleDelConnections(data, cell);
            } else {
                CellConnectionProcessor::scheduleDelCell(data, cell, index);
            }
        }
    }
}

