#include "DataAccessKernels.cuh"

namespace
{
    __device__ void copyString(int& targetLen, int& targetStringIndex, int sourceLen, char* sourceString, int& numStringBytes, char*& stringBytes)
    {
        targetLen = sourceLen;
        if (sourceLen > 0) {
            targetStringIndex = atomicAdd(&numStringBytes, sourceLen);
            for (int i = 0; i < sourceLen; ++i) {
                stringBytes[targetStringIndex + i] = sourceString[i];
            }
        }
    }

    __device__ void createCellTO(Cell* cell, DataAccessTO& dataTO, Cell* cellArrayStart)
    {
        auto cellTOIndex = atomicAdd(dataTO.numCells, 1);
        auto& cellTO = dataTO.cells[cellTOIndex];

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
            cellTO.metadata.nameLen, cellTO.metadata.nameStringIndex, cell->metadata.nameLen, cell->metadata.name, *dataTO.numStringBytes, dataTO.stringBytes);
        copyString(
            cellTO.metadata.descriptionLen,
            cellTO.metadata.descriptionStringIndex,
            cell->metadata.descriptionLen,
            cell->metadata.description,
            *dataTO.numStringBytes,
            dataTO.stringBytes);
        copyString(
            cellTO.metadata.sourceCodeLen,
            cellTO.metadata.sourceCodeStringIndex,
            cell->metadata.sourceCodeLen,
            cell->metadata.sourceCode,
            *dataTO.numStringBytes,
            dataTO.stringBytes);

        cell->tag = cellTOIndex;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;
            cellTO.connections[i].cellIndex = connectingCell - cellArrayStart;
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

    __device__ void createParticleTO(Particle* particle, DataAccessTO& dataTO)
    {
        int particleTOIndex = atomicAdd(dataTO.numParticles, 1);
        ParticleAccessTO& particleTO = dataTO.particles[particleTOIndex];

        particleTO.id = particle->id;
        particleTO.pos = particle->absPos;
        particleTO.vel = particle->vel;
        particleTO.energy = particle->energy;
    }

}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/
__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, DataAccessTO dataTO)
{
    auto const& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.entities.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if ((includeClusters && cell->selected == 0) || (!includeClusters && cell->selected != 1)) {
            cell->tag = -1;
            continue;
        }
        createCellTO(cell, dataTO, cellArrayStart);
    }
}

__global__ void cudaGetSelectedParticleData(SimulationData data, DataAccessTO access)
{
    PartitionData particleBlock = calcPartition(data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.entities.particlePointers.at(particleIndex);
        if (particle->selected == 0) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, DataAccessTO dataTO)
{
    auto const& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.entities.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        bool found = false;
        for (int i = 0; i < Const::MaxInspectedEntities; ++i) {
            if (ids.values[i] == 0) {
                break;
            }
            if (ids.values[i] == cell->id) {
                found = true;
            }
        }
        if (!found) {
            cell->tag = -1;
            continue;
        }

        createCellTO(cell, dataTO, cellArrayStart);
    }
}

__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, DataAccessTO access)
{
    PartitionData particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.entities.particlePointers.at(particleIndex);
        bool found = false;
        for (int i = 0; i < Const::MaxInspectedEntities; ++i) {
            if (ids.values[i] == 0) {
                break;
            }
            if (ids.values[i] == particle->id) {
                found = true;
            }
        }
        if (!found) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO)
{
    {
        auto const& cells = data.entities.cellPointers;
        auto const partition = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);

            auto pos = cell->absPos;
            data.cellMap.mapPosCorrection(pos);
            if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
                continue;
            }
            auto cellTOIndex = atomicAdd(dataTO.numCells, 1);
            auto& cellTO = dataTO.cells[cellTOIndex];

            cellTO.id = cell->id;
            cellTO.pos = cell->absPos;
            cellTO.cellFunctionType = cell->cellFunctionType;
            cellTO.selected = cell->selected;
            cellTO.branchNumber = cell->branchNumber;
        }
    }
    {
        auto const& particles = data.entities.particlePointers;
        auto const partition = calcAllThreadsPartition(particles.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);

            auto pos = particle->absPos;
            data.particleMap.mapPosCorrection(pos);
            if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
                continue;
            }
            auto particleTOIndex = atomicAdd(dataTO.numParticles, 1);
            auto& particleTO = dataTO.particles[particleTOIndex];

            particleTO.id = particle->id;
            particleTO.pos = particle->absPos;
            particleTO.selected = particle->selected;
        }
    }
}

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO)
{
    auto const& cells = data.entities.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const cellArrayStart = data.entities.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto pos = cell->absPos;
        data.cellMap.mapPosCorrection(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            cell->tag = -1;
            continue;
        }

        createCellTO(cell, dataTO, cellArrayStart);
    }
}

__global__ void cudaResolveConnections(SimulationData data, DataAccessTO dataTO)
{
    auto const partition = calcAllThreadsPartition(*dataTO.numCells);
    auto const firstCell = data.entities.cells.getArray();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cellTO = dataTO.cells[index];

        for (int i = 0; i < cellTO.numConnections; ++i) {
            auto const cellIndex = cellTO.connections[i].cellIndex;
            cellTO.connections[i].cellIndex = data.entities.cells.at(cellIndex).tag;
        }
    }
}

__global__ void cudaGetTokenData(SimulationData data, DataAccessTO dataTO)
{
    auto const& tokens = data.entities.tokenPointers;

    auto partition = calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (auto tokenIndex = partition.startIndex; tokenIndex <= partition.endIndex; ++tokenIndex) {
        auto token = tokens.at(tokenIndex);

        if (token->cell->tag == -1) {
            continue;
        }

        auto tokenTOIndex = atomicAdd(dataTO.numTokens, 1);
        auto& tokenTO = dataTO.tokens[tokenTOIndex];

        tokenTO.energy = token->energy;
        for (int i = 0; i < cudaSimulationParameters.tokenMemorySize; ++i) {
            tokenTO.memory[i] = token->memory[i];
        }
        tokenTO.cellIndex = token->cell->tag;
        tokenTO.sequenceNumber = tokenIndex;
    }
}

__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access)
{
    PartitionData particleBlock = calcPartition(data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        auto const& particle = data.entities.particlePointers.at(particleIndex);
        auto pos = particle->absPos;
        data.particleMap.mapPosCorrection(pos);
        if (!isContainedInRect(rectUpperLeft, rectLowerRight, pos)) {
            continue;
        }

        createParticleTO(particle, access);
    }
}

__global__ void cudaCreateDataFromTO(SimulationData data, DataAccessTO dataTO, bool selectNewData, bool createIds)
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(&data);
    }
    __syncthreads();

    auto particlePartition = calcPartition(*dataTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto particle = factory.createParticleFromTO(dataTO.particles[index], createIds);
        if (selectNewData) {
            particle->selected = 1;
        }
    }

    auto cellPartition = calcPartition(*dataTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    auto cellTargetArray = data.entities.cells.getArray() + data.entities.cells.getNumOrigEntries();
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto cell = factory.createCellFromTO(index, dataTO.cells[index], cellTargetArray, &dataTO, createIds);
        if (selectNewData) {
            cell->selected = 1;
        }
    }

    auto tokenPartition = calcPartition(*dataTO.numTokens, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = tokenPartition.startIndex; index <= tokenPartition.endIndex; ++index) {
        factory.createTokenFromTO(dataTO.tokens[index], cellTargetArray);
    }
}

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, DataAccessTO dataTO)
{
    {
        auto const partition = calcPartition(*dataTO.numCells, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = dataTO.cells[index];
            numberGen.adaptMaxId(cell.id);
        }
    }
    {
        auto const partition = calcPartition(*dataTO.numParticles, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = dataTO.particles[index];
            numberGen.adaptMaxId(particle.id);
        }
    }
}

__global__ void cudaClearDataTO(DataAccessTO dataTO)
{
    *dataTO.numCells = 0;
    *dataTO.numParticles = 0;
    *dataTO.numTokens = 0;
    *dataTO.numStringBytes = 0;
}

__global__ void cudaClearData(SimulationData data)
{
    data.entities.cellPointers.reset();
    data.entities.tokenPointers.reset();
    data.entities.particlePointers.reset();
    data.entities.cells.reset();
    data.entities.tokens.reset();
    data.entities.particles.reset();
    data.entities.dynamicMemory.reset();
}

__global__ void cudaSaveNumEntries(SimulationData data)
{
    data.entities.saveNumEntries();
}
