#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityBuilder.cuh"

#include "SimulationData.cuh"

__device__ void getClusterAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData const& data, DataAccessTO& simulationTO, int clusterIndex)
{
    Cluster const& cluster = data.clustersAC1.getEntireArray()[clusterIndex];

    int startCellIndex;
    int endCellIndex;
    calcPartition(cluster.numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

    __shared__ bool containedInRect;
    __shared__ BasicMap map;
    if (0 == threadIdx.x) {
        containedInRect = false;
        map.init(data.size);
    }
    __syncthreads();

    for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
        auto pos = cluster.cells[cellIndex].absPos;
        map.mapPosCorrection(pos);
        if (isContained(rectUpperLeft, rectLowerRight, pos)) {
            containedInRect = true;
        }
    }
    __syncthreads();

    if (containedInRect) {
        __shared__ int cellTOIndex;
        __shared__ int tokenTOIndex;
        __shared__ CellAccessTO* cellTOs;
        __shared__ TokenAccessTO* tokenTOs;
        if (0 == threadIdx.x) {
            int clusterAccessIndex = atomicAdd(simulationTO.numClusters, 1);
            cellTOIndex = atomicAdd(simulationTO.numCells, cluster.numCells);
            cellTOs = &simulationTO.cells[cellTOIndex];

            tokenTOIndex = atomicAdd(simulationTO.numTokens, cluster.numTokens);
            tokenTOs = &simulationTO.tokens[tokenTOIndex];

            ClusterAccessTO& clusterTO = simulationTO.clusters[clusterAccessIndex];
            clusterTO.id = cluster.id;
            clusterTO.pos = cluster.pos;
            clusterTO.vel = cluster.vel;
            clusterTO.angle = cluster.angle;
            clusterTO.angularVel = cluster.angularVel;
            clusterTO.numCells = cluster.numCells;
            clusterTO.numTokens = cluster.numTokens;
            clusterTO.cellStartIndex = cellTOIndex;
            clusterTO.tokenStartIndex = tokenTOIndex;
        }
        __syncthreads();

        for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
            Cell const& cell = cluster.cells[cellIndex];
            CellAccessTO& cellTO = cellTOs[cellIndex];
            cellTO.id = cell.id;
            cellTO.pos = cell.absPos;
            cellTO.energy = cell.energy;
            cellTO.maxConnections = cell.maxConnections;
            cellTO.numConnections = cell.numConnections;
            cellTO.branchNumber = cell.branchNumber;
            cellTO.tokenBlocked = cell.tokenBlocked;
            for (int i = 0; i < cell.numConnections; ++i) {
                int connectingCellIndex = cell.connections[i] - cluster.cells + cellTOIndex;
                cellTO.connectionIndices[i] = connectingCellIndex;
            }
        }

        int startTokenIndex;
        int endTokenIndex;
        calcPartition(cluster.numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);
        for (auto tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
            Token const& token = cluster.tokens[tokenIndex];
            TokenAccessTO& tokenTO = tokenTOs[tokenIndex];
            tokenTO.energy = token.energy;
            for (int i = 0; i < MAX_TOKEN_MEM_SIZE; ++i) {
                tokenTO.memory[i] = token.memory[i];
            }
            int tokenCellIndex = token.cell - cluster.cells + cellTOIndex;
            tokenTO.cellIndex = tokenCellIndex;
        }
    }
}

__device__ void getParticleAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData const& data, DataAccessTO& access, int particleIndex)
{
    Particle const& particle = data.particlesAC1.getEntireArray()[particleIndex];
    if (particle.pos.x >= rectUpperLeft.x
        && particle.pos.x <= rectLowerRight.x
        && particle.pos.y >= rectUpperLeft.y
        && particle.pos.y <= rectLowerRight.y)
    {
        int particleAccessIndex = atomicAdd(access.numParticles, 1);
        ParticleAccessTO& particleAccess = access.particles[particleAccessIndex];

        particleAccess.id = particle.id;
        particleAccess.pos = particle.pos;
        particleAccess.vel = particle.vel;
        particleAccess.energy = particle.energy;
    }
}

__global__ void getSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    int indexResource = blockIdx.x;
    int numEntities = data.clustersAC1.getNumEntries();
    int startIndex;
    int endIndex;
    calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);

    for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
        getClusterAccessData(rectUpperLeft, rectLowerRight, data, access, clusterIndex);
    }

    indexResource = threadIdx.x + blockIdx.x * blockDim.x;
    numEntities = data.particlesAC1.getNumEntries();

    calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
    for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
        getParticleAccessData(rectUpperLeft, rectLowerRight, data, access, particleIndex);
    }

}

__device__ void filterClusterData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData& data, int clusterIndex)
{
    Cluster const& cluster = data.clustersAC1.getEntireArray()[clusterIndex];

    int startCellIndex;
    int endCellIndex;
    calcPartition(cluster.numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

    __shared__ bool containedInRect;
    if (0 == threadIdx.x) {
        containedInRect = false;
    }
    __syncthreads();

    for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
        Cell const& cell = cluster.cells[cellIndex];
        if (isContained(rectUpperLeft, rectLowerRight, cell.absPos)) {
            containedInRect = true;
        }
    }
    __syncthreads();

    if (!containedInRect) {
        __shared__ Cell* newCells;
        __shared__ Cluster* newCluster;
        if (0 == threadIdx.x) {
            newCluster = data.clustersAC2.getNewElement();
            newCells = data.cellsAC2.getNewSubarray(cluster.numCells);
            *newCluster = cluster;
            newCluster->cells = newCells;
        }
        __syncthreads();

        for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
            Cell& cell = cluster.cells[cellIndex];
            newCells[cellIndex] = cell;
            newCells[cellIndex].cluster = newCluster;
            cell.nextTimestep = &newCells[cellIndex];
        }
        __syncthreads();

        for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
            Cell& newCell = newCells[cellIndex];
            int numConnections = newCell.numConnections;
            for (int i = 0; i < numConnections; ++i) {
                newCell.connections[i] = newCell.connections[i]->nextTimestep;
            }
        }
    }
}

__device__ void filterParticleData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData& data, int particleIndex)
{
    Particle const& particle = data.particlesAC1.getEntireArray()[particleIndex];
    if (!isContained(rectUpperLeft, rectLowerRight, particle.pos)) {
        Particle* newParticle = data.particlesAC2.getNewElement();
        *newParticle = particle;
    }
}

__global__ void filterData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
    int indexResource = blockIdx.x;
    int numEntities = data.clustersAC1.getNumEntries();
    int startIndex;
    int endIndex;
    calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);
    for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
        filterClusterData(rectUpperLeft, rectLowerRight, data, clusterIndex);
    }

    indexResource = threadIdx.x + blockIdx.x * blockDim.x;
    numEntities = data.particlesAC1.getNumEntries();
    calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
    for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
        filterParticleData(rectUpperLeft, rectLowerRight, data, particleIndex);
    }
    __syncthreads();
}


__device__ void convertData(SimulationData data, DataAccessTO const& simulationTO)
{
    __shared__ EntityBuilder converter;
    if (0 == threadIdx.x) {
        converter.init(&data, &simulationTO);
    }
    __syncthreads();

    int indexResource = blockIdx.x;
    int numEntities = *simulationTO.numClusters;
    int startIndex;
    int endIndex;
    calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);

    for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
        converter.createClusterFromTO(simulationTO.clusters[clusterIndex]);
    }

    indexResource = threadIdx.x + blockIdx.x * blockDim.x;
    numEntities = *simulationTO.numParticles;
    calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
    for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
        converter.createParticleFromTO(simulationTO.particles[particleIndex]);
    }
}

__global__ void setSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    convertData(data, access);
}
