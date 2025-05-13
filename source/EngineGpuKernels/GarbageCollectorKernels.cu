#include "GarbageCollectorKernels.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data)
{
    data.tempObjects.particlePointers.reset();
    data.tempObjects.cellPointers.reset();
}

__global__ void cudaPrepareHeapForCleanup(SimulationData data)
{
    data.tempObjects.heap.reset();
}

__global__ void cudaCleanupCellsStep1(Array<Cell*> cellPointers, Heap newHeap)
{
    //assumes that cellPointers are already cleaned up
    PartitionData cellPartition = calcAllThreadsPartition(cellPointers.getNumEntries());

    int numCellsToCopy = cellPartition.numElements();
    if (numCellsToCopy > 0) {
        auto newCells = newHeap.getTypedSubArray<Cell>(numCellsToCopy);
        auto newHeapStart = newHeap.getArray();

        int newCellIndex = 0;
        for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
            auto& cellPointer = cellPointers.at(index);
            auto& newCell = newCells[newCellIndex];
            newCell = *cellPointer;

            cellPointer->tag = reinterpret_cast<uint8_t*>(&newCell) - newHeapStart;  //save index of new cell in old cell
            cellPointer = &newCell;

            ++newCellIndex;
        }
    }
}

__global__ void cudaCleanupCellsStep2(Array<Cell*> cellPointers, Heap newHeap)
{
    {
        auto partition = calcAllThreadsPartition(cellPointers.getNumEntries());
        auto newHeapStart = newHeap.getArray();
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cellPointers.at(index);
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& connectedCell = cell->connections[i].cell;
                connectedCell = reinterpret_cast<Cell*>(newHeapStart + connectedCell->tag);
            }
        }
    }
}

namespace
{
    __device__ void copyAndAssignNewHeapData(uint8_t*& source, uint64_t numBytes, Heap& target)
    {
        if (numBytes > 0) {
            uint8_t* bytes = target.getRawSubArray(numBytes);
            for (uint64_t i = 0; i < numBytes; ++i) {
                bytes[i] = source[i];
            }
            source = bytes;
        }
    }
}

__global__ void cudaCleanupDependentCellData(Array<Cell*> cellPointers, Heap newHeap)
{
    auto const partition = calcAllThreadsPartition(cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cellPointers.at(index);
        copyAndAssignNewHeapData(cell->metadata.name, cell->metadata.nameSize, newHeap);
        copyAndAssignNewHeapData(cell->metadata.description, cell->metadata.descriptionSize, newHeap);
        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            copyAndAssignNewHeapData(
                reinterpret_cast<uint8_t*&>(cell->neuralNetwork), sizeof(*cell->neuralNetwork), newHeap);
        }
        if (cell->cellType == CellType_Constructor) {
            copyAndAssignNewHeapData(cell->cellTypeData.constructor.genome, cell->cellTypeData.constructor.genomeSize, newHeap);
        }
        if (cell->cellType == CellType_Injector) {
            copyAndAssignNewHeapData(cell->cellTypeData.injector.genome, cell->cellTypeData.injector.genomeSize, newHeap);
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
    data.objects.particlePointers.swapContent(data.tempObjects.particlePointers);
    data.objects.cellPointers.swapContent(data.tempObjects.cellPointers);
}

__global__ void cudaSwapHeaps(SimulationData data)
{
    data.objects.heap.swapContent(data.tempObjects.heap);
}


__global__ void cudaCleanupParticles(Array<Particle*> particlePointers, Heap rawMemory)
{
    //assumes that particlePointers are already cleaned up
    auto partition = calcAllThreadsPartition(particlePointers.getNumEntries());

    int numParticlesToCopy = partition.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = rawMemory.getTypedSubArray<Particle>(numParticlesToCopy);

        int newParticleIndex = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
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
    if (data.objects.heap.getNumEntries() > data.objects.heap.getSize() * Const::ArrayFillPercentage) {
        *result = true;
    } else {
        *result = false;
    }
}
