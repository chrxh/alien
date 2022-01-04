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
public:
    __inline__ __device__ void init(SimulationData* data);
    __inline__ __device__ Particle*
    createParticleFromTO(int targetIndex, ParticleAccessTO const& particleTO, Particle* particleTargetArray);
    __inline__ __device__ Cell*
    createCellFromTO(int targetIndex, CellAccessTO const& cellTO, Cell* cellArray, DataAccessTO* simulationTO);
    __inline__ __device__ void changeCellFromTO(CellAccessTO const& cellTO, DataAccessTO const& dataTO, Cell* cell);
    __inline__ __device__ Token*
    createTokenFromTO(int targetIndex, TokenAccessTO const& tokenTO, Cell* cellArray, Token* tokenArray);
    __inline__ __device__ void changeParticleFromTO(ParticleAccessTO const& particleTO, Particle* particle);
    __inline__ __device__ Particle*
    createParticle(
        float energy,
        float2 const& pos,
        float2 const& vel,
        ParticleMetadata const& metadata);
    __inline__ __device__ Cell* createRandomCell(float energy, float2 const& pos, float2 const& vel);
    __inline__ __device__ Cell* createCell();
    __inline__ __device__ Token* duplicateToken(Cell* targetCell, Token* sourceToken);
    __inline__ __device__ Token* createToken(Cell* cell, Cell* sourceCell);

private:
    __inline__ __device__ void
    copyString(int& targetLen, char*& targetString, int sourceLen, int sourceStringIndex, char* stringBytes);

    MapInfo _map;
    SimulationData* _data;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void EntityFactory::init(SimulationData* data)
{
    _data = data;
    _map.init(data->size);
}

__inline__ __device__ Particle*
EntityFactory::createParticleFromTO(int targetIndex, ParticleAccessTO const& particleTO, Particle* particleTargetArray)
{
    Particle** particlePointer = _data->entities.particlePointers.getNewElement();
    Particle* particle = particleTargetArray + targetIndex;
    *particlePointer = particle;
    
    particle->id = _data->numberGen.createNewId_kernel();
    particle->absPos = particleTO.pos;
    _map.mapPosCorrection(particle->absPos);
    particle->vel = particleTO.vel;
    particle->energy = particleTO.energy;
    particle->locked = 0;
    particle->selected = 0;
    particle->metadata.color = particleTO.metadata.color;
    return particle;
}

__inline__ __device__ Cell*
EntityFactory::createCellFromTO(int targetIndex, CellAccessTO const& cellTO, Cell* cellTargetArray, DataAccessTO* simulationTO)
{
    Cell** cellPointer = _data->entities.cellPointers.getNewElement();
    Cell* cell = cellTargetArray + targetIndex;
    *cellPointer = cell;

    cell->id = _data->numberGen.createNewId_kernel();
    cell->absPos = cellTO.pos;
    _map.mapPosCorrection(cell->absPos);
    cell->vel = cellTO.vel;
    cell->branchNumber = cellTO.branchNumber;
    cell->tokenBlocked = cellTO.tokenBlocked;
    cell->maxConnections = cellTO.maxConnections;
    cell->numConnections = cellTO.numConnections;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectingCell = cell->connections[i];
        connectingCell.cell = cellTargetArray + cellTO.connections[i].cellIndex;
        connectingCell.distance = cellTO.connections[i].distance;
        connectingCell.angleFromPrevious = cellTO.connections[i].angleFromPrevious;
    }
    cell->energy = cellTO.energy;
    cell->cellFunctionType = cellTO.cellFunctionType;

    switch (cell->cellFunctionType) {
    case Enums::CellFunction::COMPUTATION: {
        cell->numStaticBytes = cellTO.numStaticBytes;
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
        cell->staticData[i] = cellTO.staticData[i];
    }
    for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
        cell->mutableData[i] = cellTO.mutableData[i];
    }
    cell->tokenUsages = cellTO.tokenUsages;
    cell->metadata.color = cellTO.metadata.color;

    copyString(
        cell->metadata.nameLen,
        cell->metadata.name,
        cellTO.metadata.nameLen,
        cellTO.metadata.nameStringIndex,
        simulationTO->stringBytes);

    copyString(
        cell->metadata.descriptionLen,
        cell->metadata.description,
        cellTO.metadata.descriptionLen,
        cellTO.metadata.descriptionStringIndex,
        simulationTO->stringBytes);

    copyString(
        cell->metadata.sourceCodeLen,
        cell->metadata.sourceCode,
        cellTO.metadata.sourceCodeLen,
        cellTO.metadata.sourceCodeStringIndex,
        simulationTO->stringBytes);

    cell->selected = 0;
    cell->locked = 0;
    cell->temp3 = {0, 0};

    return cell;
}

__inline__ __device__ void EntityFactory::changeCellFromTO(
    CellAccessTO const& cellTO, DataAccessTO const& dataTO, Cell* cell)
{
    cell->id = cellTO.id;
    cell->absPos = cellTO.pos;
    _map.mapPosCorrection(cell->absPos);
    cell->vel = cellTO.vel;
    cell->branchNumber = cellTO.branchNumber;
    cell->tokenBlocked = cellTO.tokenBlocked;
    cell->maxConnections = cellTO.maxConnections;
    cell->energy = cellTO.energy;
    cell->cellFunctionType = cellTO.cellFunctionType;

    switch (cell->cellFunctionType) {
    case Enums::CellFunction::COMPUTATION: {
        cell->numStaticBytes = cellTO.numStaticBytes;
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
        cell->staticData[i] = cellTO.staticData[i];
    }
    for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
        cell->mutableData[i] = cellTO.mutableData[i];
    }
    cell->metadata.color = cellTO.metadata.color;

    copyString(
        cell->metadata.nameLen,
        cell->metadata.name,
        cellTO.metadata.nameLen,
        cellTO.metadata.nameStringIndex,
        dataTO.stringBytes);

    copyString(
        cell->metadata.descriptionLen,
        cell->metadata.description,
        cellTO.metadata.descriptionLen,
        cellTO.metadata.descriptionStringIndex,
        dataTO.stringBytes);

    copyString(
        cell->metadata.sourceCodeLen,
        cell->metadata.sourceCode,
        cellTO.metadata.sourceCodeLen,
        cellTO.metadata.sourceCodeStringIndex,
        dataTO.stringBytes);
}

__inline__ __device__ Token* EntityFactory::createTokenFromTO(
    int targetIndex,
    TokenAccessTO const& tokenTO,
    Cell* cellArray,
    Token* tokenArray)
{
    Token** tokenPointer = _data->entities.tokenPointers.getNewElement();
    Token* token = tokenArray + targetIndex;
    *tokenPointer = token;

    token->energy = tokenTO.energy;
    for (int i = 0; i < cudaSimulationParameters.tokenMemorySize; ++i) {
        token->memory[i] = tokenTO.memory[i];
    }
    token->cell = cellArray + tokenTO.cellIndex;
    token->sourceCell = token->cell;
    return token;
}

__inline__ __device__ void EntityFactory::changeParticleFromTO(ParticleAccessTO const& particleTO, Particle* particle)
{
    particle->energy = particleTO.energy;
    particle->absPos = particleTO.pos;
    particle->metadata.color = particleTO.metadata.color;
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
    particle->selected = 0;
    particle->locked = 0;
    particle->energy = energy;
    particle->absPos = pos;
    particle->vel = vel;
    particle->metadata = metadata;
    return particle;
}

__inline__ __device__ Cell* EntityFactory::createRandomCell(float energy, float2 const& pos, float2 const& vel)
{
    auto cell = _data->entities.cells.getNewElement();
    auto cellPointers = _data->entities.cellPointers.getNewElement();
    *cellPointers = cell;

    cell->id = _data->numberGen.createNewId_kernel();
    cell->absPos = pos;
    cell->vel = vel;
    cell->energy = energy;
    cell->maxConnections = _data->numberGen.random(MAX_CELL_BONDS);
    cell->branchNumber = _data->numberGen.random(cudaSimulationParameters.cellMaxTokenBranchNumber - 1);
    cell->numConnections = 0;
    cell->tokenBlocked = false;
    cell->locked = 0;
    cell->selected = 0;
    cell->temp3 = {0, 0};
    cell->metadata.color = 0;
    cell->metadata.nameLen = 0;
    cell->metadata.descriptionLen = 0;
    cell->metadata.sourceCodeLen = 0;
    cell->cellFunctionType = _data->numberGen.random(static_cast<int>(Enums::CellFunction::_COUNTER) - 1);
    switch (cell->cellFunctionType) {
    case Enums::CellFunction::COMPUTATION: {
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
    return cell;
}

__inline__ __device__ Cell* EntityFactory::createCell()
{
    auto result = _data->entities.cells.getNewElement();
    auto cellPointer = _data->entities.cellPointers.getNewElement();
    *cellPointer = result;
    result->tokenUsages = 0;
    result->id = _data->numberGen.createNewId_kernel();
    result->selected = 0;
    result->locked = 0;
    result->temp3 = {0, 0};
    result->metadata.color = 0;
    result->metadata.nameLen = 0;
    result->metadata.descriptionLen = 0;
    result->metadata.sourceCodeLen = 0;
    return result;
}

__inline__ __device__ Token* EntityFactory::duplicateToken(Cell* targetCell, Token* sourceToken)
{
    Token* token = _data->entities.tokens.getNewElement();
    Token** tokenPointer = _data->entities.tokenPointers.getNewElement();
    *tokenPointer = token;

    *token = *sourceToken;
    token->memory[0] = targetCell->branchNumber;
    token->sourceCell = token->cell;
    token->cell = targetCell;
    return token;
}

__inline__ __device__ Token* EntityFactory::createToken(Cell* cell, Cell* sourceCell)
{
    Token* token = _data->entities.tokens.getNewSubarray(1);
    Token** tokenPointer = _data->entities.tokenPointers.getNewElement();
    *tokenPointer = token;

    token->cell = cell;
    token->sourceCell = sourceCell;
    token->memory[0] = cell->branchNumber;
    for (int i = 1; i < MAX_TOKEN_MEM_SIZE; ++i) {
        token->memory[i] = 0;
    }
    return token;
}
