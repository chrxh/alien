#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"

#include "SimulationData.cuh"

__device__ void getClusterAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData const& data, DataAccessTO& simulationTO, int clusterIndex)
{
    auto const& cluster = data.clusterPointers.at(clusterIndex);

    BlockData cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);

    __shared__ bool containedInRect;
    __shared__ BasicMap map;
    if (0 == threadIdx.x) {
        containedInRect = false;
        map.init(data.size);
    }
    __syncthreads();

    for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        auto pos = cluster->cellPointers[cellIndex]->absPos;
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
            cellTOIndex = atomicAdd(simulationTO.numCells, cluster->numCellPointers);
            cellTOs = &simulationTO.cells[cellTOIndex];

            tokenTOIndex = atomicAdd(simulationTO.numTokens, cluster->numTokenPointers);
            tokenTOs = &simulationTO.tokens[tokenTOIndex];

            ClusterAccessTO& clusterTO = simulationTO.clusters[clusterAccessIndex];
            clusterTO.id = cluster->id;
            clusterTO.pos = cluster->pos;
            clusterTO.vel = cluster->vel;
            clusterTO.angle = cluster->angle;
            clusterTO.angularVel = cluster->angularVel;
            clusterTO.numCells = cluster->numCellPointers;
            clusterTO.numTokens = cluster->numTokenPointers;
            clusterTO.cellStartIndex = cellTOIndex;
            clusterTO.tokenStartIndex = tokenTOIndex;
        }
        __syncthreads();

        cluster->tagCellByIndex_blockCall(cellBlock);

        for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            Cell const& cell = *cluster->cellPointers[cellIndex];
            CellAccessTO& cellTO = cellTOs[cellIndex];
            cellTO.id = cell.id;
            cellTO.pos = cell.absPos;
            cellTO.energy = cell.energy;
            cellTO.maxConnections = cell.maxConnections;
            cellTO.numConnections = cell.numConnections;
            cellTO.branchNumber = cell.branchNumber;
            cellTO.tokenBlocked = cell.tokenBlocked;
            cellTO.cellFunctionType = cell.cellFunctionType;
            cellTO.numStaticBytes = cell.numStaticBytes;
            for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
                cellTO.staticData[i] = cell.staticData[i];
            }
            cellTO.numMutableBytes = cell.numMutableBytes;
            for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
                cellTO.mutableData[i] = cell.mutableData[i];
            }
            for (int i = 0; i < cell.numConnections; ++i) {
                int connectingCellIndex = cell.connections[i]->tag + cellTOIndex;
                cellTO.connectionIndices[i] = connectingCellIndex;
            }
        }

        BlockData tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            Token const& token = *cluster->tokenPointers[tokenIndex];
            TokenAccessTO& tokenTO = tokenTOs[tokenIndex];
            tokenTO.energy = token.energy;
            for (int i = 0; i < cudaSimulationParameters.tokenMemorySize; ++i) {
                tokenTO.memory[i] = token.memory[i];
            }
            int tokenCellIndex = token.cell->tag + cellTOIndex;
            tokenTO.cellIndex = tokenCellIndex;
        }
    }
}

__device__ void getParticleAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData const& data, DataAccessTO& access, int particleIndex)
{
    Particle const& particle = data.particles.getEntireArray()[particleIndex];
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
    BlockData clusterBlock = 
        calcPartition(data.clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        getClusterAccessData(rectUpperLeft, rectLowerRight, data, access, clusterIndex);
    }

    BlockData particleBlock =
        calcPartition(data.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        getParticleAccessData(rectUpperLeft, rectLowerRight, data, access, particleIndex);
    }

}

__device__ void filterClusterData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData& data, int clusterIndex)
{
    auto& cluster = data.clusterPointers.at(clusterIndex);

    BlockData cellBlock =
        calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);

    __shared__ bool containedInRect;
    if (0 == threadIdx.x) {
        containedInRect = false;
    }
    __syncthreads();

    for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        Cell const& cell = *cluster->cellPointers[cellIndex];
        if (isContained(rectUpperLeft, rectLowerRight, cell.absPos)) {
            containedInRect = true;
        }
    }
    __syncthreads();

    if (containedInRect && 0 == threadIdx.x) {
        cluster = nullptr;
    }
    __syncthreads();
}

__device__ void filterParticleData(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData& data, int particleIndex)
{
    Particle const& particle = data.particles.getEntireArray()[particleIndex];
    if (!isContained(rectUpperLeft, rectLowerRight, particle.pos)) {
        Particle* newParticle = data.particlesNew.getNewElement();
        *newParticle = particle;
    }
}

__global__ void filterData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        filterClusterData(rectUpperLeft, rectLowerRight, data, clusterIndex);
    }

    BlockData particleBlock = 
        calcPartition(data.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        filterParticleData(rectUpperLeft, rectLowerRight, data, particleIndex);
    }
    __syncthreads();
}


__device__ void convertData(SimulationData data, DataAccessTO const& simulationTO)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    BlockData clusterBlock = calcPartition(*simulationTO.numClusters, blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        factory.createClusterFromTO_blockCall(simulationTO.clusters[clusterIndex], &simulationTO);
    }

    BlockData particleBlock =
        calcPartition(*simulationTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        factory.createParticleFromTO(simulationTO.particles[particleIndex], &simulationTO);
    }
}

__global__ void setSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    convertData(data, access);
}
