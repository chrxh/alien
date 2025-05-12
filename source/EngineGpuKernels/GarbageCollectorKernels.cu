#include "GarbageCollectorKernels.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data)
{
    data.tempObjects.particlePointers.reset();
    data.tempObjects.cellPointers.reset();
}

__global__ void cudaPrepareArraysForCleanup(SimulationData data)
{
    data.tempObjects.rawMemory.reset();
}

__global__ void cudaCleanupCellsStep1(Array<Cell*> cellPointers, RawMemory rawMemory)
{
    //assumes that cellPointers are already cleaned up
    PartitionData cellPartition = calcAllThreadsPartition(cellPointers.getNumEntries());

    int numCellsToCopy = cellPartition.numElements();
    if (numCellsToCopy > 0) {
        auto newCells = rawMemory.getTypedSubArray<Cell>(numCellsToCopy);

        int newCellIndex = 0;
        for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
            auto& cellPointer = cellPointers.at(index);
            auto& newCell = newCells[newCellIndex];
            newCell = *cellPointer;

            cellPointer->tag = reinterpret_cast<uint8_t*>(&newCell) - rawMemory.getArray();  //save index of new cell in old cell
            cellPointer = &newCell;

            ++newCellIndex;
        }
    }
}

__global__ void cudaCleanupCellsStep2(Array<Cell*> cellPointers, RawMemory rawMemory)
{
    {
        auto partition = calcAllThreadsPartition(cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cellPointers.at(index);
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& connectedCell = cell->connections[i].cell;
                cell->connections[i].cell = reinterpret_cast<Cell*>(rawMemory.getArray() + connectedCell->tag);
            }
        }
    }
}

namespace
{
    __device__ void copyAndAssignNewAuxiliaryData(uint8_t*& source, uint64_t numBytes, RawMemory& target)
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

__global__ void cudaCleanupRawMemory(Array<Cell*> cellPointers, RawMemory auxiliaryData)
{
    auto const partition = calcAllThreadsPartition(cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cellPointers.at(index);
        copyAndAssignNewAuxiliaryData(cell->metadata.name, cell->metadata.nameSize, auxiliaryData);
        copyAndAssignNewAuxiliaryData(cell->metadata.description, cell->metadata.descriptionSize, auxiliaryData);
        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            copyAndAssignNewAuxiliaryData(
                reinterpret_cast<uint8_t*&>(cell->neuralNetwork), sizeof(*cell->neuralNetwork), auxiliaryData);
        }
        if (cell->cellType == CellType_Constructor) {
            copyAndAssignNewAuxiliaryData(cell->cellTypeData.constructor.genome, cell->cellTypeData.constructor.genomeSize, auxiliaryData);
        }
        if (cell->cellType == CellType_Injector) {
            copyAndAssignNewAuxiliaryData(cell->cellTypeData.injector.genome, cell->cellTypeData.injector.genomeSize, auxiliaryData);
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

__global__ void cudaSwapRawMemory(SimulationData data)
{
    data.objects.rawMemory.swapContent(data.tempObjects.rawMemory);
}


__global__ void cudaCleanupParticles(Array<Particle*> particlePointers, RawMemory rawMemory)
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
    if (data.objects.rawMemory.getNumEntries() > data.objects.rawMemory.getSize() * Const::ArrayFillLevelFactor) {
        *result = true;
    } else {
        *result = false;
    }
}
