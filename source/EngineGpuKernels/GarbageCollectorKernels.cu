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
    data.tempObjects.auxiliaryData.reset();
}

__global__ void cudaCleanupCellsStep1(Array<Cell*> cellPointers, Array<Cell> cells)
{
    //assumes that cellPointers are already cleaned up
    PartitionData pointerBlock = calcPartition(cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numCellsToCopy = pointerBlock.numElements();
    if (numCellsToCopy > 0) {
        auto newCells = cells.getSubArray(numCellsToCopy);

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
    __device__ void copyAndAssignNewAuxiliaryData(uint8_t*& source, uint64_t numBytes, RawMemory& target)
    {
        if (numBytes > 0) {
            uint8_t* bytes = target.getAlignedSubArray(numBytes);
            for (uint64_t i = 0; i < numBytes; ++i) {
                bytes[i] = source[i];
            }
            source = bytes;
        }
    }
}

__global__ void cudaCleanupAuxiliaryData(Array<Cell*> cellPointers, RawMemory auxiliaryData)
{
    auto const partition = calcAllThreadsPartition(cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cellPointers.at(index);
        copyAndAssignNewAuxiliaryData(cell->metadata.name, cell->metadata.nameSize, auxiliaryData);
        copyAndAssignNewAuxiliaryData(cell->metadata.description, cell->metadata.descriptionSize, auxiliaryData);
        switch (cell->cellFunction) {
        case Enums::CellFunction_Neuron:
            copyAndAssignNewAuxiliaryData(
                reinterpret_cast<uint8_t*&>(cell->cellFunctionData.neuron.neuronState),
                sizeof(*cell->cellFunctionData.neuron.neuronState),
                auxiliaryData);
            break;
        case Enums::CellFunction_Constructor:
            copyAndAssignNewAuxiliaryData(
                cell->cellFunctionData.constructor.genome, cell->cellFunctionData.constructor.genomeSize, auxiliaryData);
            break;
        case Enums::CellFunction_Injector:
            copyAndAssignNewAuxiliaryData(
                cell->cellFunctionData.injector.genome, cell->cellFunctionData.injector.genomeSize, auxiliaryData);
            break;
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

__global__ void cudaSwapArrays(SimulationData data)
{
    data.objects.cells.swapContent(data.tempObjects.cells);
    data.objects.particles.swapContent(data.tempObjects.particles);
    data.objects.auxiliaryData.swapContent(data.tempObjects.auxiliaryData);
}


__global__ void cudaCleanupParticles(Array<Particle*> particlePointers, Array<Particle> particles)
{
    //assumes that particlePointers are already cleaned up
    PartitionData pointerBlock = calcPartition(particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numParticlesToCopy = pointerBlock.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = particles.getSubArray(numParticlesToCopy);

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
