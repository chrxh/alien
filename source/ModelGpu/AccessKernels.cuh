#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "CleanupKernels.cuh"

#include "SimulationData.cuh"

__global__ void getClusterAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO simulationTO, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    PartitionData clusterBlock =
        calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {

        auto const& cluster = clusters.at(clusterIndex);

        PartitionData cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);

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

            PartitionData tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
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
}

__global__ void getParticleAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    PartitionData particleBlock =
        calcPartition(data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto& particle = *data.entities.particlePointers.at(particleIndex);
        if (particle.absPos.x >= rectUpperLeft.x
            && particle.absPos.x <= rectLowerRight.x
            && particle.absPos.y >= rectUpperLeft.y
            && particle.absPos.y <= rectLowerRight.y)
        {
            int particleAccessIndex = atomicAdd(access.numParticles, 1);
            ParticleAccessTO& particleAccess = access.particles[particleAccessIndex];

            particleAccess.id = particle.id;
            particleAccess.pos = particle.absPos;
            particleAccess.vel = particle.vel;
            particleAccess.energy = particle.energy;
        }
    }
}

__device__ void filterCluster(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData& data, int clusterArrayIndex, int clusterIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    auto& cluster = clusters.at(clusterIndex);

    PartitionData cellBlock =
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

__device__ void filterParticle(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    SimulationData& data, int particleIndex)
{
    auto& particle = data.entities.particlePointers.getEntireArray()[particleIndex];
    if (isContained(rectUpperLeft, rectLowerRight, particle->absPos)) {
        particle = nullptr;
    }
}

__global__ void filterClusters(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);

    PartitionData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        filterCluster(rectUpperLeft, rectLowerRight, data, clusterArrayIndex, clusterIndex);
    }
    __syncthreads();
}

__global__ void filterParticles(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
    PartitionData particleBlock =
        calcPartition(data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        filterParticle(rectUpperLeft, rectLowerRight, data, particleIndex);
    }
    __syncthreads();
}


__global__ void convertData(SimulationData data, DataAccessTO simulationTO)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    PartitionData clusterBlock = calcPartition(*simulationTO.numClusters, blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        factory.createClusterFromTO_blockCall(simulationTO.clusters[clusterIndex], &simulationTO);
    }

    PartitionData particleBlock =
        calcPartition(*simulationTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        factory.createParticleFromTO(simulationTO.particles[particleIndex]);
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void getSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    *access.numClusters = 0;
    *access.numCells = 0;
    *access.numParticles = 0;
    *access.numTokens = 0;

    MULTI_CALL(getClusterAccessData, rectUpperLeft, rectLowerRight, data, access);
    cudaDeviceSynchronize();
    getParticleAccessData << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> >(rectUpperLeft, rectLowerRight, data, access);
    cudaDeviceSynchronize();
}

__global__ void setSimulationAccessData(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    MULTI_CALL(filterClusters, rectUpperLeft, rectLowerRight, data);
    filterParticles<< <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (rectUpperLeft, rectLowerRight, data);
    cudaDeviceSynchronize();
    convertData << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data, access);
    cudaDeviceSynchronize();

    data.cellMap.reset();
    data.particleMap.reset();
    cleanup<<<1, 1>>>(data);
    cudaDeviceSynchronize();
}
