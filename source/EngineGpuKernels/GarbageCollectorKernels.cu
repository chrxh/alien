#include "GarbageCollectorKernels.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data)
{
    data.tempObjects.particlePointers.reset();
    data.tempObjects.cellPointers.reset();
}

__global__ void cudaPrepareArraysForCleanup(SimulationData data)
{
    data.tempObjects.particles.reset();
    data.tempObjects.cells.reset();
    data.tempObjects.additionalData.reset();
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

__global__ void cudaCleanupCellsStep2(Array<Cell> cells)
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
}

namespace
{
    __device__ void copyBytes(char*& string, int numBytes, RawMemory& stringBytes)
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

__global__ void cudaCleanupRawBytes(Array<Cell*> cellPointers, RawMemory stringBytes)
{
    auto const partition = calcAllThreadsPartition(cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cellPointers.at(index);
        copyBytes(cell->metadata.name, cell->metadata.nameSize, stringBytes);
        copyBytes(cell->metadata.description, cell->metadata.descriptionSize, stringBytes);
        copyBytes(cell->metadata.sourceCode, cell->metadata.sourceCodeLen, stringBytes);
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
    data.objects.particlePointers.swapContent(data.tempObjects.particlePointers);
    data.objects.cellPointers.swapContent(data.tempObjects.cellPointers);
}

__global__ void cudaSwapArrays(SimulationData data)
{
    data.objects.cells.swapContent(data.tempObjects.cells);
    data.objects.particles.swapContent(data.tempObjects.particles);
    data.objects.additionalData.swapContent(data.tempObjects.additionalData);
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
    if (data.objects.particles.getNumEntries() > data.objects.particles.getSize() * Const::ArrayFillLevelFactor
        || data.objects.cells.getNumEntries() > data.objects.cells.getSize() * Const::ArrayFillLevelFactor) {
        *result = true;
    } else {
        *result = false;
    }
}
