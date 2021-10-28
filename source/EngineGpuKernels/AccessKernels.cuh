#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "CleanupKernels.cuh"

#include "SimulationData.cuh"

#define SELECTION_RADIUS 30

__device__ void copyString(
    int& targetLen,
    int& targetStringIndex,
    int sourceLen,
    char* sourceString,
    int& numStringBytes,
    char*& stringBytes)
{
    targetLen = sourceLen;
    if (sourceLen > 0) {
        targetStringIndex = atomicAdd(&numStringBytes, sourceLen);
        for (int i = 0; i < sourceLen; ++i) {
            stringBytes[targetStringIndex + i] = sourceString[i];
        }
    }
}

__global__ void tagCells(int2 rectUpperLeft, int2 rectLowerRight, Array<Cell*> cells)
{
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, cell->absPos)) {
            cell->tag = 1;
        } else {
            cell->tag = 0;
        }
    }
}

__global__ void rolloutTagToCellClusters(Array<Cell*> cells, int* change)
{
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        auto tag = atomicAdd(&cell->tag, 0);
        if (1 == tag) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& connectingCell = cell->connections[i].cell;
                auto origTag = atomicExch(&connectingCell->tag, 1);
                if (0 == origTag) {
                    atomicExch(change, 1);
                }
            }
        }
    }
}

__global__ void tagClusters(int2 rectUpperLeft, int2 rectLowerRight, Array<Cell*> cells)
{
    KERNEL_CALL(tagCells, rectUpperLeft, rectLowerRight, cells);
    int* change = new int;
    do {
        *change = 0;
        KERNEL_CALL(rolloutTagToCellClusters, cells, change);
    } while (1 == *change);
}

__global__ void deleteTaggedCells(Array<Cell*> cells)
{
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (1 == cell->tag) {
            cell = nullptr;
        }
    }
}

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ void getCellAccessDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO accessTO)
{
    auto const& cells = data.entities.cellPointers;
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    auto const firstCell = data.entities.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (0 == cell->tag) {
            continue;
        }

        auto pos = cell->absPos;
        data.cellMap.mapPosCorrection(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }
        auto cellTOIndex = atomicAdd(accessTO.numCells, 1);
        auto& cellTO = accessTO.cells[cellTOIndex];

        cellTO.id = cell->id;
        cellTO.pos = cell->absPos;
        cellTO.vel = cell->vel;
        cellTO.energy = cell->energy;
        cellTO.maxConnections = cell->maxConnections;
        cellTO.numConnections = cell->numConnections;
        cellTO.branchNumber = cell->branchNumber;
        cellTO.tokenBlocked = cell->tokenBlocked;
        cellTO.cellFunctionType = cell->cellFunctionType;
        cellTO.numStaticBytes = cell->numStaticBytes;
        cellTO.tokenUsages = cell->tokenUsages;
        cellTO.metadata.color = cell->metadata.color;

        copyString(
            cellTO.metadata.nameLen,
            cellTO.metadata.nameStringIndex,
            cell->metadata.nameLen,
            cell->metadata.name,
            *accessTO.numStringBytes,
            accessTO.stringBytes);
        copyString(
            cellTO.metadata.descriptionLen,
            cellTO.metadata.descriptionStringIndex,
            cell->metadata.descriptionLen,
            cell->metadata.description,
            *accessTO.numStringBytes,
            accessTO.stringBytes);
        copyString(
            cellTO.metadata.sourceCodeLen,
            cellTO.metadata.sourceCodeStringIndex,
            cell->metadata.sourceCodeLen,
            cell->metadata.sourceCode,
            *accessTO.numStringBytes,
            accessTO.stringBytes);
        cell->tag = cellTOIndex;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;
            cellTO.connections[i].cellIndex = connectingCell - firstCell;
            cellTO.connections[i].distance = cell->connections[i].distance;
            cellTO.connections[i].angleFromPrevious = cell->connections[i].angleFromPrevious;
        }
        for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
            cellTO.staticData[i] = cell->staticData[i];
        }
        cellTO.numMutableBytes = cell->numMutableBytes;
        for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
            cellTO.mutableData[i] = cell->mutableData[i];
        }
    }

/*
            PartitionData tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
            for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
                Token const& token = *cluster->tokenPointers[tokenIndex];
                TokenAccessTO& tokenTO = tokenTOs[tokenIndex];
                tokenTO.energy = token.getEnergy();
                for (int i = 0; i < cudaSimulationParameters.tokenMemorySize; ++i) {
                    tokenTO.memory[i] = token.memory[i];
                }
                int tokenCellIndex = token.cell->tag + cellTOIndex;
                tokenTO.cellIndex = tokenCellIndex;
            }
*/
}

__global__ void resolveConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO accessTO)
{
    auto const partition =
        calcPartition(*accessTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    auto const firstCell = data.entities.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cellTO = accessTO.cells[index];
        
        for (int i = 0; i < cellTO.numConnections; ++i) {
            auto const cellIndex = cellTO.connections[i].cellIndex;
            cellTO.connections[i].cellIndex = data.entities.cells.at(cellIndex).tag;
        }
    }
}

__global__ void getCellAccessData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO accessTO)
{
    KERNEL_CALL_1_1(tagClusters, rectUpperLeft, rectLowerRight, data.entities.cellPointers);
    KERNEL_CALL(getCellAccessDataWithoutConnections, rectUpperLeft, rectLowerRight, data, accessTO);
    KERNEL_CALL(resolveConnections, rectUpperLeft, rectLowerRight, data, accessTO);
}

__global__ void getTokenAccessData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO accessTO)
{
    auto const& tokens = data.entities.tokenPointers;

    auto partition =
        calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (auto tokenIndex = partition.startIndex; tokenIndex <= partition.endIndex; ++tokenIndex) {
        auto token = tokens.at(tokenIndex);

        auto tokenTOIndex = atomicAdd(accessTO.numTokens, 1);
        auto& tokenTO = accessTO.tokens[tokenTOIndex];

        tokenTO.energy = token->energy;
        for (int i = 0; i < cudaSimulationParameters.tokenMemorySize; ++i) {
            tokenTO.memory[i] = token->memory[i];
        }
        tokenTO.cellIndex = token->cell->tag;
    }
}

__global__ void getParticleAccessData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access)
{
    PartitionData particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = *data.entities.particlePointers.at(particleIndex);
        auto pos = particle.absPos;
        data.particleMap.mapPosCorrection(pos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, particle.absPos)) {
            int particleAccessIndex = atomicAdd(access.numParticles, 1);
            ParticleAccessTO& particleAccess = access.particles[particleAccessIndex];

            particleAccess.id = particle.id;
            particleAccess.pos = particle.absPos;
            particleAccess.vel = particle.vel;
            particleAccess.energy = particle.energy;
        }
    }
}

__global__ void filterCells(int2 rectUpperLeft, int2 rectLowerRight, Array<Cell*> cells)
{
    KERNEL_CALL_1_1(tagClusters, rectUpperLeft, rectLowerRight, cells);
    KERNEL_CALL_1_1(deleteTaggedCells, cells);
}

__device__ void filterParticle(int2 const& rectUpperLeft, int2 const& rectLowerRight,
    Array<Particle*> particles, int particleIndex)
{
    auto& particle = particles.getArray()[particleIndex];
    if (isContainedInRect(rectUpperLeft, rectLowerRight, particle->absPos)) {
        particle = nullptr;
    }
}

__global__ void filterParticles(int2 rectUpperLeft, int2 rectLowerRight, Array<Particle*> particles)
{
    PartitionData particleBlock =
        calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        filterParticle(rectUpperLeft, rectLowerRight, particles, particleIndex);
    }
    __syncthreads();
}


__global__ void createDataFromTO(
    SimulationData data,
    DataAccessTO simulationTO,
    Particle* particleTargetArray,
    Cell* cellTargetArray,
    Token* tokenTargetArray)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition =
        calcPartition(*simulationTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        factory.createParticleFromTO(index, simulationTO.particles[index], particleTargetArray);
    }

    auto cellPartition =
        calcPartition(*simulationTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        factory.createCellFromTO(index, simulationTO.cells[index], cellTargetArray, &simulationTO);
    }

    auto tokenPartition =
        calcPartition(*simulationTO.numTokens, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = tokenPartition.startIndex; index <= tokenPartition.endIndex; ++index) {
        factory.createTokenFromTO(index, simulationTO.tokens[index], cellTargetArray, tokenTargetArray, &simulationTO);
    }
}

__global__ void selectParticles(int2 pos, Array<Particle*> particles)
{
    auto const particleBlock = calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);

        if (Math::lengthSquared(particle->absPos - pos) < SELECTION_RADIUS) {
            particle->setSelected(true);
        }
    }
}

__global__ void deselectParticles(Array<Particle*> particles)
{
    auto const particleBlock = calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);
        particle->setSelected(false);
    }
}

__global__ void adaptNumberGenerator(CudaNumberGenerator numberGen, DataAccessTO accessTO)
{
    {
        auto const partition =
            calcPartition(*accessTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = accessTO.cells[index];
            numberGen.adaptMaxId(cell.id);
        }
    }
    {
        auto const partition =
            calcPartition(*accessTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = accessTO.particles[index];
            numberGen.adaptMaxId(particle.id);
        }
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/
__global__ void getSimulationAccessDataKernel(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    *access.numCells = 0;
    *access.numParticles = 0;
    *access.numTokens = 0;
    *access.numStringBytes = 0;

    KERNEL_CALL_1_1(getCellAccessData, rectUpperLeft, rectLowerRight, data, access);
    KERNEL_CALL(getTokenAccessData, rectUpperLeft, rectLowerRight, data, access);
    KERNEL_CALL(getParticleAccessData, rectUpperLeft, rectLowerRight, data, access);
}

__global__ void setSimulationAccessDataKernel(int2 rectUpperLeft, int2 rectLowerRight,
    SimulationData data, DataAccessTO access)
{
    KERNEL_CALL(adaptNumberGenerator, data.numberGen, access);
/*
    KERNEL_CALL_1_1(filterCells, {0, 0}, {0, 0}, data.entities.cellPointers);
    KERNEL_CALL(filterParticles, {0, 0}, {0, 0}, data.entities.particlePointers);
*/
    KERNEL_CALL_1_1(filterCells, rectUpperLeft, rectLowerRight, data.entities.cellPointers);
    KERNEL_CALL(filterParticles, rectUpperLeft, rectLowerRight, data.entities.particlePointers);
    KERNEL_CALL(
        createDataFromTO,
        data,
        access,
        data.entities.particles.getNewSubarray(*access.numParticles),
        data.entities.cells.getNewSubarray(*access.numCells),
        data.entities.tokens.getNewSubarray(*access.numTokens));
    KERNEL_CALL_1_1(cleanupAfterDataManipulationKernel, data);
}

__global__ void cudaClearData(SimulationData data)
{
    data.entities.cellPointers.reset();
    data.entities.tokenPointers.reset();
    data.entities.particlePointers.reset();
    data.entities.cells.reset();
    data.entities.tokens.reset();
    data.entities.particles.reset();
}

__global__ void cudaSelectData(int2 pos, SimulationData data)
{
/*
    KERNEL_CALL(selectClusters, pos, data.entities.clusterPointers);
*/
    KERNEL_CALL(selectParticles, pos, data.entities.particlePointers);
}

__global__ void cudaDeselectData(SimulationData data)
{
/*
    KERNEL_CALL(deselectClusters, data.entities.clusterPointers);
*/
    KERNEL_CALL(deselectParticles, data.entities.particlePointers);
}