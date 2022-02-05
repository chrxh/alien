#include "GarbageCollectorKernels.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data)
{
    data.entitiesForCleanup.particlePointers.reset();
    data.entitiesForCleanup.cellPointers.reset();
    data.entitiesForCleanup.tokenPointers.reset();
}

__global__ void cudaPrepareArraysForCleanup(SimulationData data)
{
    data.entitiesForCleanup.particles.reset();
    data.entitiesForCleanup.cells.reset();
    data.entitiesForCleanup.tokens.reset();
    data.entitiesForCleanup.dynamicMemory.reset();
}

__global__ void cudaCleanupCellsStep1(Array<Cell*> cellPointers, Array<Cell> cells)
{
    //assumes that cellPointers are already cleaned up
    PartitionData pointerBlock = calcPartition(cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numCellsToCopy = pointerBlock.numElements();
    if (numCellsToCopy > 0) {
        auto newCells = cells.getNewSubarray(numCellsToCopy);

        int newCellIndex = 0;
        for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
            auto& cellPointer = cellPointers.at(index);
            auto& newCell = newCells[newCellIndex];
            newCell = *cellPointer;

            cellPointer->tag = &newCell - cells.getArray();  //save index of new cell in old cell
            cellPointer = &newCell;

            ++newCellIndex;
        }
    }
}

__global__ void cudaCleanupCellsStep2(Array<Token*> tokenPointers, Array<Cell> cells)
{
    {
        auto partition = calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            for (int i = 0; i < cell.numConnections; ++i) {
                auto& connectedCell = cell.connections[i].cell;
                cell.connections[i].cell = &cells.at(connectedCell->tag);
            }
        }
    }
    {
        auto partition = calcPartition(tokenPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            if (auto& token = tokenPointers.at(index)) {
                token->cell = &cells.at(token->cell->tag);
                token->sourceCell = &cells.at(token->sourceCell->tag);
            }
        }
    }
}

__global__ void cudaCleanupTokens(Array<Token*> tokenPointers, Array<Token> newToken)
{
    auto partition = calcPartition(tokenPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    if (partition.numElements() > 0) {
        Token* newEntities = newToken.getNewSubarray(partition.numElements());

        int targetIndex = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& token = tokenPointers.at(index);
            newEntities[targetIndex] = *token;
            token = &newEntities[targetIndex];
            ++targetIndex;
        }
    }
}

namespace
{
    __device__ void copyString(char*& string, int numBytes, TempMemory& stringBytes)
    {
        if (numBytes > 0) {
            char* newString = stringBytes.getArray<char>(numBytes);
            for (int i = 0; i < numBytes; ++i) {
                newString[i] = string[i];
            }
            string = newString;
        }
    }
}

__global__ void cudaCleanupStringBytes(Array<Cell*> cellPointers, TempMemory stringBytes)
{
    auto const partition = calcAllThreadsPartition(cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cellPointers.at(index);

/*
        if (cell->metadata.sourceCodeLen > 0) {
            printf("num: %d\n", cell->metadata.sourceCodeLen);
            for (int i = 0; i < cell->metadata.sourceCodeLen; ++i) {
                printf("%c", cell->metadata.sourceCode[i]);
            }
            printf("+\n");
        }
*/
        copyString(cell->metadata.name, cell->metadata.nameLen, stringBytes);
        copyString(cell->metadata.description, cell->metadata.descriptionLen, stringBytes);
        copyString(cell->metadata.sourceCode, cell->metadata.sourceCodeLen, stringBytes);
    }
}

__global__ void cudaCleanupStringBytes2(Array<Cell*> cellPointers)
{
    auto const partition = calcAllThreadsPartition(cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cellPointers.at(index);
        if(!cell) {
            printf("OHOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n");
        }
        if (cell->metadata.sourceCodeLen > 0) {
//            printf("num: %d\n", cell->metadata.sourceCodeLen);
            for (int i = 0; i < cell->metadata.sourceCodeLen; ++i) {
                cell->metadata.sourceCode[i] = cell->metadata.sourceCode[i] +1;
//                printf("%c|", cell->metadata.sourceCode[i]);
            }
//            printf("+\n");
        }
    }
}

__global__ void cudaCleanupCellMap(SimulationData data)
{
    data.cellMap.cleanup_system();
}

__global__ void cudaCleanupParticleMap(SimulationData data)
{
    data.particleMap.cleanup_system();
}

__global__ void cudaSwapPointerArrays(SimulationData data)
{
    data.entities.particlePointers.swapContent(data.entitiesForCleanup.particlePointers);
    data.entities.cellPointers.swapContent(data.entitiesForCleanup.cellPointers);
    data.entities.tokenPointers.swapContent(data.entitiesForCleanup.tokenPointers);
}

__global__ void cudaSwapArrays(SimulationData data)
{
    data.entities.cells.swapContent(data.entitiesForCleanup.cells);
    data.entities.tokens.swapContent(data.entitiesForCleanup.tokens);
    data.entities.particles.swapContent(data.entitiesForCleanup.particles);
    data.entities.dynamicMemory.swapContent(data.entitiesForCleanup.dynamicMemory);
}


__global__ void cudaCleanupParticles(Array<Particle*> particlePointers, Array<Particle> particles)
{
    //assumes that particlePointers are already cleaned up
    PartitionData pointerBlock = calcPartition(particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numParticlesToCopy = pointerBlock.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = particles.getNewSubarray(numParticlesToCopy);

        int newParticleIndex = 0;
        for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
            auto& particlePointer = particlePointers.at(index);
            auto& newParticle = newParticles[newParticleIndex];
            newParticle = *particlePointer;
            particlePointer = &newParticle;

            ++newParticleIndex;
        }
    }
}

__global__ void cudaCheckIfCleanupIsNecessary(SimulationData data, bool* result)
{
    if (data.entities.particles.getNumEntries() > data.entities.particles.getSize() * Const::ArrayFillLevelFactor
        || data.entities.cells.getNumEntries() > data.entities.cells.getSize() * Const::ArrayFillLevelFactor
        || data.entities.tokens.getNumEntries() > data.entities.tokens.getSize() * Const::ArrayFillLevelFactor) {
        *result = true;
    } else {
        *result = false;
    }
}
