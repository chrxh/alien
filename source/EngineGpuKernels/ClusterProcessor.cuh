#pragma once

#include "Entities.cuh"
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
        if (object->type != ObjectType_Solid) {
            continue;
        }
        object->typeData.solid.clusterIndex = index;
        object->typeData.solid.clusterBoundaries = 0;
        object->typeData.solid.clusterPos = {0, 0};
        object->typeData.solid.clusterVel = {0, 0};
        object->typeData.solid.clusterAngularMass = 0;
        object->typeData.solid.clusterAngularMomentum = 0;
        object->typeData.solid.numCellsInCluster = 0;
    }
}

__device__ __inline__ void ClusterProcessor::findClusterIteration(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto currentObject = objects.at(index);
        if (currentObject->type != ObjectType_Solid) {
            continue;
        }

        //heuristics to cover connected cells
        for (int i = 0; i < 30; ++i) {
            bool found = false;
            for (int j = 0; j < currentObject->numConnections; ++j) {
                auto candidateObject = currentObject->connections[j].object;
                if (candidateObject->type != ObjectType_Solid) {
                    continue;
                }
                auto cellTag = currentObject->typeData.solid.clusterIndex;
                auto origTag = atomicMin(&candidateObject->typeData.solid.clusterIndex, cellTag);
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
        if (object->type != ObjectType_Solid) {
            continue;
        }
        auto cluster = objects.at(object->typeData.solid.clusterIndex);
        if (object->pos.x < data.worldSize.x / 3) {
            atomicOr(&cluster->typeData.solid.clusterBoundaries, 1);
        }
        if (object->pos.y < data.worldSize.y / 3) {
            atomicOr(&cluster->typeData.solid.clusterBoundaries, 2);
        }
    }
}

__device__ __inline__ void ClusterProcessor::accumulateClusterPosAndVel(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->type != ObjectType_Solid) {
            continue;
        }
        auto cluster = objects.at(object->typeData.solid.clusterIndex);
        atomicAdd(&cluster->typeData.solid.clusterVel.x, object->vel.x);
        atomicAdd(&cluster->typeData.solid.clusterVel.y, object->vel.y);

        //topology correction
        auto cellPos = object->pos;
        if ((cluster->typeData.solid.clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->typeData.solid.clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }

        atomicAdd(&cluster->typeData.solid.clusterPos.x, cellPos.x);
        atomicAdd(&cluster->typeData.solid.clusterPos.y, cellPos.y);

        atomicAdd(&cluster->typeData.solid.numCellsInCluster, 1);
    }
}

__device__ __inline__ void ClusterProcessor::accumulateClusterAngularProp(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->type != ObjectType_Solid) {
            continue;
        }
        auto cluster = objects.at(object->typeData.solid.clusterIndex);
        auto clusterVel = cluster->typeData.solid.clusterVel / cluster->typeData.solid.numCellsInCluster;
        auto clusterPos = cluster->typeData.solid.clusterPos / cluster->typeData.solid.numCellsInCluster;

        //topology correction
        auto cellPos = object->pos;
        if ((cluster->typeData.solid.clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->typeData.solid.clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularMass = Math::lengthSquared(r);
        auto angularMomentum = Physics::angularMomentum(r, object->vel - clusterVel);
        atomicAdd(&cluster->typeData.solid.clusterAngularMass, angularMass);
        atomicAdd(&cluster->typeData.solid.clusterAngularMomentum, angularMomentum);
    }
}

__device__ __inline__ void ClusterProcessor::applyClusterData(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = objects.at(index);
        if (object->type != ObjectType_Solid) {
            continue;
        }
        auto cluster = objects.at(object->typeData.solid.clusterIndex);
        auto clusterPos = cluster->typeData.solid.clusterPos / cluster->typeData.solid.numCellsInCluster;
        auto clusterVel = cluster->typeData.solid.clusterVel / cluster->typeData.solid.numCellsInCluster;

        auto cellPos = object->pos;
        if ((cluster->typeData.solid.clusterBoundaries & 1) == 1 && cellPos.x > data.worldSize.x * 2 / 3) {
            cellPos.x -= data.worldSize.x;
        }
        if ((cluster->typeData.solid.clusterBoundaries & 2) == 2 && cellPos.y > data.worldSize.y * 2 / 3) {
            cellPos.y -= data.worldSize.y;
        }
        auto r = cellPos - clusterPos;

        auto angularVel = Physics::angularVelocity(cluster->typeData.solid.clusterAngularMomentum, cluster->typeData.solid.clusterAngularMass);

        auto rigidity = ParameterCalculator::calcParameter(cudaSimulationParameters.rigidity, data, object->pos) * object->stiffness * object->stiffness;
        object->vel = object->vel * (1.0f - rigidity) + Physics::tangentialVelocity(r, clusterVel, angularVel) * rigidity;
    }
}
