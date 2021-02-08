#pragma once

#include "Base.cuh"
#include "AccessTOs.cuh"
#include "Map.cuh"
#include "Math.cuh"
#include "EngineInterface/ElementaryTypes.h"
#include "Particle.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

class EntityFactory
{
private:
    MapInfo _map;
    SimulationData* _data;

public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Cell* createCell(Cluster* cluster);
    __inline__ __device__ Token* createToken(Cell* cell, Cell* sourceCell);
    __inline__ __device__ Particle* createParticleFromTO(
        ParticleAccessTO const& particleTO);  //TODO: not adding to simulation!
    __inline__ __device__ Particle* createParticle(
        float energy,
        float2 const& pos,
        float2 const& vel,
        ParticleMetadata const& metadata);  //TODO: not adding to simulation!
    __inline__ __device__ Cluster* createCluster(Cluster** clusterPointerToReuse = nullptr);
    __inline__ __device__ void createClusterFromTO_block(
        ClusterAccessTO const& clusterTO,
        DataAccessTO const* _simulationTO);
    __inline__ __device__ Cluster* createClusterWithRandomCell(float energy, float2 const& pos, float2 const& vel);

private:
    __inline__ __device__ void
    copyString(int& targetLen, char*& targetString, int sourceLen, int sourceStringIndex, char* stringBytes);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void EntityFactory::init(SimulationData* data)
{
    _data = data;
    _map.init(data->size);
}

__inline__ __device__ Cluster * EntityFactory::createCluster(Cluster** clusterPointerToReuse)
{
    auto result = _data->entities.clusters.getNewElement();
    result->metadata.nameLen = 0;
    result->init();

    auto newClusterPointer = clusterPointerToReuse ? clusterPointerToReuse : _data->entities.clusterPointers.getNewElement();
    *newClusterPointer = result;
    return result;
}

__inline__ __device__ void EntityFactory::createClusterFromTO_block(
    ClusterAccessTO const& clusterTO,
    DataAccessTO const* simulationTO)
{
    __shared__ Cluster* cluster;
    __shared__ Cell* cells;
    __shared__ Token* tokens;
    __shared__ float angularMass;
    __shared__ float invRotMatrix[2][2];
    __shared__ float2 posCorrection;

    if (0 == threadIdx.x) {
        auto clusterPointer = _data->entities.clusterPointers.getNewElement();
        cluster = _data->entities.clusters.getNewElement();
        *clusterPointer = cluster;
        cluster->id = clusterTO.id;
        cluster->pos = clusterTO.pos;
        _map.mapPosCorrection(cluster->pos);
        posCorrection = cluster->pos - clusterTO.pos;
        cluster->setVelocity(clusterTO.vel);
        cluster->setSelected(false);
        cluster->angle = clusterTO.angle;
        cluster->setAngularVelocity(clusterTO.angularVel);
        cluster->numCellPointers = clusterTO.numCells;
        cluster->cellPointers = _data->entities.cellPointers.getNewSubarray(cluster->numCellPointers);
        cells = _data->entities.cells.getNewSubarray(cluster->numCellPointers);
        cluster->numTokenPointers = clusterTO.numTokens;
        cluster->tokenPointers = _data->entities.tokenPointers.getNewSubarray(cluster->numTokenPointers);
        tokens = _data->entities.tokens.getNewSubarray(cluster->numTokenPointers);

        copyString(
            cluster->metadata.nameLen,
            cluster->metadata.name,
            clusterTO.metadata.nameLen,
            clusterTO.metadata.nameStringIndex,
            simulationTO->stringBytes);

        cluster->decompositionRequired = 0;
        cluster->locked = 0;
        cluster->clusterToFuse = nullptr;
        cluster->init();

        angularMass = 0.0f;
        Math::inverseRotationMatrix(cluster->angle, invRotMatrix);
    }
    __syncthreads();

    PartitionData cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
    for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        auto& cell = cells[cellIndex];
        cluster->cellPointers[cellIndex] = &cell;
        auto const& cellTO = simulationTO->cells[clusterTO.cellStartIndex + cellIndex];
        cell.id = cellTO.id;
        cell.cluster = cluster;
        cell.absPos = cellTO.pos + posCorrection;

        float2 deltaPos = cell.absPos - clusterTO.pos;
        cell.relPos.x = deltaPos.x * invRotMatrix[0][0] + deltaPos.y * invRotMatrix[0][1];
        cell.relPos.y = deltaPos.x * invRotMatrix[1][0] + deltaPos.y * invRotMatrix[1][1];
        atomicAdd(&angularMass, Math::lengthSquared(cell.relPos));

        auto r = cell.absPos - cluster->pos;
        _map.mapDisplacementCorrection(r);
        cell.vel = Physics::tangentialVelocity(r, cluster->getVelocity(), cluster->getAngularVelocity());

        cell.setEnergy_safe(cellTO.energy);
        cell.branchNumber = cellTO.branchNumber;
        cell.tokenBlocked = cellTO.tokenBlocked;
        cell.setFused(false);
        cell.maxConnections = cellTO.maxConnections;
        cell.numConnections = cellTO.numConnections;
        for (int i = 0; i < cell.numConnections; ++i) {
            int index = cellTO.connectionIndices[i] - clusterTO.cellStartIndex;
            cell.connections[i] = cells + index;
        }

        cell.setCellFunctionType(cellTO.cellFunctionType);

        switch (cell.getCellFunctionType()) {
        case Enums::CellFunction::COMPUTER: {
            cell.numStaticBytes = cellTO.numStaticBytes;
            cell.numMutableBytes = cudaSimulationParameters.cellFunctionComputerCellMemorySize;
        } break;
        case Enums::CellFunction::SENSOR: {
            cell.numStaticBytes = 0;
            cell.numMutableBytes = 5;
        } break;
        default: {
            cell.numStaticBytes = 0;
            cell.numMutableBytes = 0;
        }
        }
        for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
            cell.staticData[i] = cellTO.staticData[i];
        }
        for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
            cell.mutableData[i] = cellTO.mutableData[i];
        }
        cell.tokenUsages = cellTO.tokenUsages;
        cell.metadata.color = cellTO.metadata.color;

        copyString(
            cell.metadata.nameLen,
            cell.metadata.name,
            cellTO.metadata.nameLen,
            cellTO.metadata.nameStringIndex,
            simulationTO->stringBytes);

        copyString(
            cell.metadata.descriptionLen,
            cell.metadata.description,
            cellTO.metadata.descriptionLen,
            cellTO.metadata.descriptionStringIndex,
            simulationTO->stringBytes);

        copyString(
            cell.metadata.sourceCodeLen,
            cell.metadata.sourceCode,
            cellTO.metadata.sourceCodeLen,
            cellTO.metadata.sourceCodeStringIndex,
            simulationTO->stringBytes);

        cell.initProtectionCounter();
        cell.alive = 1;
        cell.locked = 0;
    }

    PartitionData tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);

    for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        auto& token = tokens[tokenIndex];
        cluster->tokenPointers[tokenIndex] = &token;
        auto const& tokenTO = simulationTO->tokens[clusterTO.tokenStartIndex + tokenIndex];

        token.setEnergy(tokenTO.energy);
        for (int i = 0; i < cudaSimulationParameters.tokenMemorySize; ++i) {
            token.memory[i] = tokenTO.memory[i];
        }
        int index = tokenTO.cellIndex - clusterTO.cellStartIndex;
        token.cell = cells + index;
        token.sourceCell = token.cell;
    }

    __syncthreads();

    if (0 == threadIdx.x) {
        cluster->angularMass = angularMass;
    }
}

__inline__ __device__ Cell* EntityFactory::createCell(Cluster* cluster)
{
    auto result = _data->entities.cells.getNewSubarray(1);
    result->cluster = cluster;
    result->tokenUsages = 0;
    result->id = _data->numberGen.createNewId_kernel();
    result->locked = 0;
    result->initProtectionCounter();
    result->alive = 1;
    result->metadata.color = 0;
    result->metadata.nameLen = 0;
    result->metadata.descriptionLen = 0;
    result->metadata.sourceCodeLen = 0;
    result->setFused(false);
    return result;
}

__inline__ __device__ Token* EntityFactory::createToken(Cell* cell, Cell* sourceCell)
{
    auto result =  _data->entities.tokens.getNewSubarray(1);
    result->cell = cell;
    result->sourceCell = sourceCell;
    result->memory[0] = cell->branchNumber;
    return result;
}

__inline__ __device__ Particle* EntityFactory::createParticleFromTO(ParticleAccessTO const& particleTO)
{
    Particle** particlePointer = _data->entities.particlePointers.getNewElement();
    Particle* particle = _data->entities.particles.getNewElement();
    *particlePointer = particle;
    particle->id = 0 == particleTO.id ? _data->numberGen.createNewId_kernel() : particleTO.id;
    particle->absPos = particleTO.pos;
    _map.mapPosCorrection(particle->absPos);
    particle->vel = particleTO.vel;
    particle->setEnergy(particleTO.energy);
    particle->locked = 0;
    particle->alive = 1;
    particle->metadata.color = particleTO.metadata.color;
    particle->setSelected(false);
    return particle;
}

__inline__ __device__ Cluster*
EntityFactory::createClusterWithRandomCell(float energy, float2 const& pos, float2 const& vel)
{
    auto clusterPointer = _data->entities.clusterPointers.getNewElement();
    auto cluster = _data->entities.clusters.getNewElement();
    *clusterPointer = cluster;

    auto cell = _data->entities.cells.getNewElement();
    auto cellPointers = _data->entities.cellPointers.getNewElement();

    cluster->id = _data->numberGen.createNewId_kernel();
    cluster->pos = pos;
    cluster->setVelocity(vel);
    cluster->setSelected(false);
    cluster->angle = 0.0f;
    cluster->setAngularVelocity(0);
    cluster->angularMass = 0.0f;
    cluster->numCellPointers = 1;
    cluster->cellPointers = cellPointers;
    *cellPointers = cell;
    cluster->numTokenPointers = 0;
    cluster->metadata.nameLen = 0;

    cluster->clusterToFuse = nullptr;
    cluster->locked = 0;
    cluster->decompositionRequired = 0;
    cluster->init();

    cell->id = _data->numberGen.createNewId_kernel();
    cell->absPos = pos;
    cell->relPos = {0.0f, 0.0f};
    cell->vel = vel;
    cell->setEnergy_safe(energy);
    cell->maxConnections = _data->numberGen.random(MAX_CELL_BONDS);
    cell->cluster = cluster;
    cell->branchNumber = _data->numberGen.random(cudaSimulationParameters.cellMaxTokenBranchNumber - 1);
    cell->numConnections = 0;
    cell->tokenBlocked = false;
    cell->alive = 1;
    cell->initProtectionCounter();
    cell->locked = 0;
    cell->setFused(false);
    cell->metadata.color = 0;
    cell->metadata.nameLen = 0;
    cell->metadata.descriptionLen = 0;
    cell->metadata.sourceCodeLen = 0;
    cell->setCellFunctionType(_data->numberGen.random(static_cast<int>(Enums::CellFunction::_COUNTER) - 1));
    switch (cell->getCellFunctionType()) {
    case Enums::CellFunction::COMPUTER: {
        cell->numStaticBytes = cudaSimulationParameters.cellFunctionComputerMaxInstructions * 3;
        cell->numMutableBytes = cudaSimulationParameters.cellFunctionComputerCellMemorySize;
    } break;
    case Enums::CellFunction::SENSOR: {
        cell->numStaticBytes = 0;
        cell->numMutableBytes = 5;
    } break;
    default: {
        cell->numStaticBytes = 0;
        cell->numMutableBytes = 0;
    }
    }
    for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
        cell->staticData[i] = _data->numberGen.random(255);
    }
    for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
        cell->mutableData[i] = _data->numberGen.random(255);
    }
    cell->tokenUsages = 0;
    return cluster;
}

__inline__ __device__ void
EntityFactory::copyString(int& targetLen, char*& targetString, int sourceLen, int sourceStringIndex, char* stringBytes)
{
    targetLen = sourceLen;
    if (sourceLen > 0) {
        targetString = _data->entities.strings.getArray<char>(sourceLen);
        for (int i = 0; i < sourceLen; ++i) {
            targetString[i] = stringBytes[sourceStringIndex + i];
        }
    }
}

__inline__ __device__ Particle*
EntityFactory::createParticle(float energy, float2 const& pos, float2 const& vel, ParticleMetadata const& metadata)
{
    Particle** particlePointer = _data->entities.particlePointers.getNewElement();
    Particle* particle = _data->entities.particles.getNewElement();
    *particlePointer = particle;
    particle->id = _data->numberGen.createNewId_kernel();
    particle->locked = 0;
    particle->alive = 1;
    particle->setEnergy(energy);
    particle->absPos = pos;
    particle->vel = vel;
    particle->metadata = metadata;
    particle->setSelected(false);
    return particle;
}
