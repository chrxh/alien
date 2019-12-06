#pragma once

#include "Base.cuh"
#include "AccessTOs.cuh"
#include "Map.cuh"
#include "Math.cuh"
#include "ModelBasic/ElementaryTypes.h"
#include "Particle.cuh"
#include "Physics.cuh"
#include "SimulationData.cuh"
#include "Token.cuh"
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

class EntityFactory
{
private:
    MapInfo _map;
    SimulationData* _data;

public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Cell* createCell(Cluster* cluster);
    __inline__ __device__ Token* createToken(Cell* cell);
    __inline__ __device__ Particle* createParticleFromTO(
        ParticleAccessTO const& particleTO);  //TODO: not adding to simulation!
    __inline__ __device__ Particle* createParticle(
        float energy,
        float2 const& pos,
        float2 const& vel,
        ParticleMetadata const& metadata);  //TODO: not adding to simulation!
    __inline__ __device__ void createClusterFromTO_blockCall(
        ClusterAccessTO const& clusterTO,
        DataAccessTO const* _simulationTO);
    __inline__ __device__ Cluster* createClusterWithRandomCell(float energy, float2 const& pos, float2 const& vel);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void EntityFactory::init(SimulationData* data)
{
    _data = data;
    _map.init(data->size);
}

__inline__ __device__ void EntityFactory::createClusterFromTO_blockCall(
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
        auto clusterPointer = _data->entities.clusterPointerArrays.getNewClusterPointer(clusterTO.numCells);
        cluster = _data->entities.clusters.getNewElement();
        *clusterPointer = cluster;
        cluster->id = clusterTO.id;
        cluster->pos = clusterTO.pos;
        _map.mapPosCorrection(cluster->pos);
        posCorrection = cluster->pos - clusterTO.pos;
        cluster->vel = clusterTO.vel;
        cluster->angle = clusterTO.angle;
        cluster->angularVel = clusterTO.angularVel;
        cluster->numCellPointers = clusterTO.numCells;
        cluster->cellPointers = _data->entities.cellPointers.getNewSubarray(cluster->numCellPointers);
        cells = _data->entities.cells.getNewSubarray(cluster->numCellPointers);
        cluster->numTokenPointers = clusterTO.numTokens;
        cluster->tokenPointers = _data->entities.tokenPointers.getNewSubarray(cluster->numTokenPointers);
        tokens = _data->entities.tokens.getNewSubarray(cluster->numTokenPointers);

        cluster->decompositionRequired = 0;
        cluster->locked = 0;
        cluster->clusterToFuse = nullptr;

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
        cell.vel = Physics::tangentialVelocity(r, cluster->vel, cluster->angularVel);

        cell.setEnergy(cellTO.energy);
        cell.branchNumber = cellTO.branchNumber;
        cell.tokenBlocked = cellTO.tokenBlocked;
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
        cell.age = cellTO.age;
        cell.metadata.color = cellTO.metadata.color;

        cell.protectionCounter = 0;
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
    result->age = 0;
    result->id = _data->numberGen.createNewId_kernel();
    result->locked = 0;
    result->protectionCounter = 0;
    result->alive = 1;
    result->metadata.color = 0;
    return result;
}

__inline__ __device__ Token* EntityFactory::createToken(Cell* cell)
{
    auto result =  _data->entities.tokens.getNewSubarray(1);
    result->cell = cell;
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
    return particle;
}

__inline__ __device__ Cluster*
EntityFactory::createClusterWithRandomCell(float energy, float2 const& pos, float2 const& vel)
{
    auto clusterPointer = _data->entities.clusterPointerArrays.getNewClusterPointer(1);
    auto cluster = _data->entities.clusters.getNewElement();
    *clusterPointer = cluster;

    auto cell = _data->entities.cells.getNewElement();
    auto cellPointers = _data->entities.cellPointers.getNewElement();

    cluster->id = _data->numberGen.createNewId_kernel();
    cluster->pos = pos;
    cluster->vel = vel;
    cluster->angle = 0.0f;
    cluster->angularVel = 0.0f;
    cluster->angularMass = 0.0f;
    cluster->numCellPointers = 1;
    cluster->cellPointers = cellPointers;
    *cellPointers = cell;
    cluster->numTokenPointers = 0;

    cluster->clusterToFuse = nullptr;
    cluster->locked = 0;
    cluster->decompositionRequired = 0;

    cell->id = _data->numberGen.createNewId_kernel();
    cell->absPos = pos;
    cell->relPos = {0.0f, 0.0f};
    cell->vel = vel;
    cell->setEnergy(energy);
    cell->maxConnections = _data->numberGen.random(MAX_CELL_BONDS);
    cell->cluster = cluster;
    cell->branchNumber = _data->numberGen.random(cudaSimulationParameters.cellMaxTokenBranchNumber - 1);
    cell->numConnections = 0;
    cell->tokenBlocked = false;
    cell->alive = 1;
    cell->protectionCounter = 0;
    cell->locked = 0;
    cell->metadata.color = 0;
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
    cell->age = 0;
    return cluster;
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
    return particle;
}
