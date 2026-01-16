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
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        object->typeData.cell.clusterIndex = index;
        object->typeData.cell.clusterBoundaries = 0;
        object->typeData.cell.clusterPos = {0, 0};
        object->typeData.cell.clusterVel = {0, 0};
        object->typeData.cell.clusterAngularMass = 0;
        object->typeData.cell.clusterAngularMomentum = 0;
        object->typeData.cell.numCellsInCluster = 0;
    }
}

__device__ __inline__ void ClusterProcessor::findClusterIteration(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto currentObject = objects.at(index);

        //heuristics to cover connected cells
        for (int i = 0; i < 30; ++i) {
            bool found = false;
            for (int j = 0; j < currentObject->numConnections; ++j) {
                auto candidateObject = currentObject->connections[j].object;
                auto cellTag = currentObject->typeData.cell.clusterIndex;
                auto origTag = atomicMin(&candidateObject->typeData.cell.clusterIndex, cellTag);
                if (cellTag < origTag) {
                    currentObject = candidateObject;
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
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        auto cluster = objects.at(object->typeData.cell.clusterIndex);
        if (object->pos.x < data.worldSize.x / 3) {
            atomicOr(&cluster->typeData.cell.clusterBoundaries, 1);
        }
        if (object->pos.y < data.worldSize.y / 3) {
            atomicOr(&cluster->typeData.cell.clusterBoundaries, 2);
        }
    }
}

__device__ __inline__ void ClusterProcessor::accumulateClusterPosAndVel(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        auto cluster = objects.at(object->typeData.cell.clusterIndex);
        atomicAdd(&cluster->typeData.cell.clusterVel.x, object->vel.x);
        atomicAdd(&cluster->typeData.cell.clusterVel.y, object->vel.y);

        //topology correction
        auto cellPos = object->pos;
        if ((cluster->typeData.cell.clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->typeData.cell.clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }

        atomicAdd(&cluster->typeData.cell.clusterPos.x, cellPos.x);
        atomicAdd(&cluster->typeData.cell.clusterPos.y, cellPos.y);

        atomicAdd(&cluster->typeData.cell.numCellsInCluster, 1);
    }
}

__device__ __inline__ void ClusterProcessor::accumulateClusterAngularProp(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        auto cluster = objects.at(object->typeData.cell.clusterIndex);
        auto clusterVel = cluster->typeData.cell.clusterVel / cluster->typeData.cell.numCellsInCluster;
        auto clusterPos = cluster->typeData.cell.clusterPos / cluster->typeData.cell.numCellsInCluster;

        //topology correction
        auto cellPos = object->pos;
        if ((cluster->typeData.cell.clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->typeData.cell.clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularMass = Math::lengthSquared(r);
        auto angularMomentum = Physics::angularMomentum(r, object->vel - clusterVel);
        atomicAdd(&cluster->typeData.cell.clusterAngularMass, angularMass);
        atomicAdd(&cluster->typeData.cell.clusterAngularMomentum, angularMomentum);
    }
}

__device__ __inline__ void ClusterProcessor::applyClusterData(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        auto cluster = objects.at(object->typeData.cell.clusterIndex);
        auto clusterPos = cluster->typeData.cell.clusterPos / cluster->typeData.cell.numCellsInCluster;
        auto clusterVel = cluster->typeData.cell.clusterVel / cluster->typeData.cell.numCellsInCluster;

        auto cellPos = object->pos;
        if ((cluster->typeData.cell.clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->typeData.cell.clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularVel = Physics::angularVelocity(cluster->typeData.cell.clusterAngularMomentum, cluster->typeData.cell.clusterAngularMass);

        auto rigidity = ParameterCalculator::calcParameter(cudaSimulationParameters.rigidity, data, object->pos) * object->stiffness * object->stiffness;
        object->vel = object->vel * (1.0f - rigidity) + Physics::tangentialVelocity(r, clusterVel, angularVel) * rigidity;
    }
}
