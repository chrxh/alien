#pragma once

#include "Entity.cuh"
#include "ParameterCalculator.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"

class ClusterProcessor
{
public:
    __device__ __inline__ static void initClusterData(SimulationData& data);
    __device__ __inline__ static void findClusterIteration(SimulationData& data);
    __device__ __inline__ static void findClusterBoundaries(SimulationData& data);
    __device__ __inline__ static void accumulateClusterPosAndVel(SimulationData& data);
    __device__ __inline__ static void accumulateClusterAngularProp(SimulationData& data);
    __device__ __inline__ static void applyClusterData(SimulationData& data);

private:
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void ClusterProcessor::initClusterData(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        object->clusterIndex = index;
        object->clusterBoundaries = 0;
        object->clusterPos = {0, 0};
        object->clusterVel = {0, 0};
        object->clusterAngularMass = 0;
        object->clusterAngularMomentum = 0;
        object->numCellsInCluster = 0;
    }
}

__device__ __inline__ void ClusterProcessor::findClusterIteration(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto currentCell = cells.at(index);

        //heuristics to cover connected cells
        for (int i = 0; i < 30; ++i) {
            bool found = false;
            for (int j = 0; j < currentCell->numConnections; ++j) {
                auto candidateCell = currentCell->connections[j].object;
                auto cellTag = currentCell->clusterIndex;
                auto origTag = atomicMin(&candidateCell->clusterIndex, cellTag);
                if (cellTag < origTag) {
                    currentCell = candidateCell;
                    found = true;
                    break;
                }
            }
            if (!found) {
                break;
            }
        }
    }
}

__device__ __inline__ void ClusterProcessor::findClusterBoundaries(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = cells.at(index);
        auto cluster = cells.at(object->clusterIndex);
        if (object->pos.x < data.worldSize.x / 3) {
            atomicOr(&cluster->clusterBoundaries, 1);
        }
        if (object->pos.y < data.worldSize.y / 3) {
            atomicOr(&cluster->clusterBoundaries, 2);
        }
    }
}

__device__ __inline__ void ClusterProcessor::accumulateClusterPosAndVel(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = cells.at(index);
        auto cluster = cells.at(object->clusterIndex);
        atomicAdd(&cluster->clusterVel.x, object->vel.x);
        atomicAdd(&cluster->clusterVel.y, object->vel.y);

        //topology correction
        auto cellPos = object->pos;
        if ((cluster->clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }

        atomicAdd(&cluster->clusterPos.x, cellPos.x);
        atomicAdd(&cluster->clusterPos.y, cellPos.y);

        atomicAdd(&cluster->numCellsInCluster, 1);
    }
}

__device__ __inline__ void ClusterProcessor::accumulateClusterAngularProp(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = cells.at(index);
        auto cluster = cells.at(object->clusterIndex);
        auto clusterVel = cluster->clusterVel / cluster->numCellsInCluster;
        auto clusterPos = cluster->clusterPos / cluster->numCellsInCluster;

        //topology correction
        auto cellPos = object->pos;
        if ((cluster->clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularMass = Math::lengthSquared(r);
        auto angularMomentum = Physics::angularMomentum(r, object->vel - clusterVel);
        atomicAdd(&cluster->clusterAngularMass, angularMass);
        atomicAdd(&cluster->clusterAngularMomentum, angularMomentum);
    }
}

__device__ __inline__ void ClusterProcessor::applyClusterData(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = cells.at(index);
        auto cluster = cells.at(object->clusterIndex);
        auto clusterPos = cluster->clusterPos / cluster->numCellsInCluster;
        auto clusterVel = cluster->clusterVel / cluster->numCellsInCluster;

        auto cellPos = object->pos;
        if ((cluster->clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularVel = Physics::angularVelocity(cluster->clusterAngularMomentum, cluster->clusterAngularMass);

        auto rigidity = ParameterCalculator::calcParameter(cudaSimulationParameters.rigidity, data, object->pos) * object->stiffness * object->stiffness;
        object->vel = object->vel * (1.0f - rigidity) + Physics::tangentialVelocity(r, clusterVel, angularVel) * rigidity;
    }
}
