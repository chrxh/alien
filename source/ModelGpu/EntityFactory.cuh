#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"

#include "SimulationData.cuh"

class EntityFactory
{
private:
    BasicMap _map;
    SimulationData* _data;

public:
    __device__ void init(SimulationData* data);
    __device__ void createClusterFromTO(ClusterAccessTO const& clusterTO, DataAccessTO const* _simulationTO);
    __device__ void createParticleFromTO(ParticleAccessTO const& particleTO, DataAccessTO const* _simulationTO);
    __device__ void createClusterWithRandomCell(float energy, float2 const & pos, float2 const & vel);

};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ void EntityFactory::init(SimulationData* data)
{
    _data = data;
    _map.init(data->size);
}

__device__ void EntityFactory::createClusterFromTO(ClusterAccessTO const& clusterTO, DataAccessTO const* _simulationTO)
{
    __shared__ Cluster* cluster;
    __shared__ float angularMass;
    __shared__ float invRotMatrix[2][2];
    __shared__ float2 posCorrection;

    if (0 == threadIdx.x) {
        cluster = _data->clustersAC2.getNewElement();
        cluster->id = clusterTO.id;
        cluster->pos = clusterTO.pos;
        _map.mapPosCorrection(cluster->pos);
        posCorrection = sub(cluster->pos, clusterTO.pos);
        cluster->vel = clusterTO.vel;
        cluster->angle = clusterTO.angle;
        cluster->angularVel = clusterTO.angularVel;
        cluster->numCells = clusterTO.numCells;
        cluster->cells = _data->cellsAC2.getNewSubarray(cluster->numCells);
        cluster->numTokens = clusterTO.numTokens;
        cluster->tokens = _data->tokensAC2.getNewSubarray(cluster->numTokens);

        cluster->decompositionRequired = false;
        cluster->locked = 0;
        cluster->clusterToFuse = nullptr;

        angularMass = 0.0f;
        Physics::inverseRotationMatrix(cluster->angle, invRotMatrix);
    }
    __syncthreads();

    int startCellIndex;
    int endCellIndex;
    calcPartition(cluster->numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

    for (auto cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
        Cell& cell = cluster->cells[cellIndex];
        CellAccessTO const& cellTO = _simulationTO->cells[clusterTO.cellStartIndex + cellIndex];
        cell.id = cellTO.id;
        cell.cluster = cluster;
        cell.absPos = add(cellTO.pos, posCorrection);

        float2 deltaPos = sub(cell.absPos, clusterTO.pos);
        cell.relPos.x = deltaPos.x*invRotMatrix[0][0] + deltaPos.y*invRotMatrix[0][1];
        cell.relPos.y = deltaPos.x*invRotMatrix[1][0] + deltaPos.y*invRotMatrix[1][1];
        atomicAdd(&angularMass, lengthSquared(cell.relPos));

        auto r = sub(cell.absPos, cluster->pos);
        _map.mapDisplacementCorrection(r);
        cell.vel = Physics::tangentialVelocity(r, cluster->vel, cluster->angularVel);

        cell.energy = cellTO.energy;
        cell.branchNumber = cellTO.branchNumber;
        cell.tokenBlocked = cellTO.tokenBlocked;
        cell.maxConnections = cellTO.maxConnections;
        cell.numConnections = cellTO.numConnections;
        for (int i = 0; i < cell.numConnections; ++i) {
            int index = cellTO.connectionIndices[i] - clusterTO.cellStartIndex;
            cell.connections[i] = cluster->cells + index;
        }

        cell.cellFunctionType = cellTO.cellFunctionType;
        cell.numStaticBytes = cellTO.numStaticBytes;
        for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
            cell.staticData[i] = cellTO.staticData[i];
        }
        cell.numMutableBytes = cellTO.numMutableBytes;
        for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
            cell.mutableData[i] = cellTO.mutableData[i];
        }

        cell.nextTimestep = nullptr;
        cell.protectionCounter = 0;
        cell.alive = true;
        cell.locked = 0;
    }

    int startTokenIndex;
    int endTokenIndex;
    calcPartition(cluster->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);

    for (auto tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
        Token& token = cluster->tokens[tokenIndex];
        TokenAccessTO const& tokenTO = _simulationTO->tokens[clusterTO.tokenStartIndex + tokenIndex];

        token.energy = tokenTO.energy;
        for (int i = 0; i < MAX_TOKEN_MEM_SIZE; ++i) {
            token.memory[i] = tokenTO.memory[i];
        }
        int index = tokenTO.cellIndex - clusterTO.cellStartIndex;
        token.cell = cluster->cells + index;
    }

    __syncthreads();

    if (0 == threadIdx.x) {
        cluster->angularMass = angularMass;
    }
}

__device__ void EntityFactory::createParticleFromTO(ParticleAccessTO const& particleTO, DataAccessTO const* _simulationTO)
{
    Particle* particle = _data->particlesAC2.getNewElement();
    particle->id = particleTO.id;
    particle->pos = particleTO.pos;
    _map.mapPosCorrection(particle->pos);
    particle->vel = particleTO.vel;
    particle->energy = particleTO.energy;
    particle->locked = 0;
    particle->alive = true;
}

__device__ void EntityFactory::createClusterWithRandomCell(float energy, float2 const & pos, float2 const & vel)
{
    Cluster* cluster = _data->clustersAC2.getNewElement();
    Cell* cell = _data->cellsAC2.getNewElement();

    cluster->id = _data->numberGen.createNewId_kernel();
    cluster->pos = pos;
    cluster->vel = vel;
    cluster->angle = 0.0f;
    cluster->angularVel = 0.0f;
    cluster->angularMass = 0.0f;
    cluster->numCells = 1;
    cluster->cells = cell;
    cluster->numTokens = 0;

    cluster->clusterToFuse = nullptr;
    cluster->locked = 0;
    cluster->decompositionRequired = false;

    cell->id = _data->numberGen.createNewId_kernel();
    cell->absPos = pos;
    cell->relPos = { 0.0f, 0.0f };
    cell->vel = vel;
    cell->energy = energy;
    cell->maxConnections = _data->numberGen.random(MAX_CELL_BONDS);
    cell->cluster = cluster;
    cell->branchNumber = _data->numberGen.random(cudaSimulationParameters.cellMaxTokenBranchNumber - 1);
    cell->numConnections = 0;
    cell->tokenBlocked = false;
    cell->nextTimestep = nullptr;
    cell->alive = true;
    cell->protectionCounter = 0;
    cell->locked = 0;
    cell->cellFunctionType = _data->numberGen.random(static_cast<int>(Enums::CellFunction::_COUNTER) - 1);
    for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
        cell->staticData[i] = _data->numberGen.random(255);
    }
    for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
        cell->mutableData[i] = _data->numberGen.random(255);
    }
}
