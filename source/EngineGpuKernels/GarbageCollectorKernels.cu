#include "GarbageCollectorKernels.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data)
{
    data.tempObjects.particles.reset();
    data.tempObjects.cells.reset();
}

__global__ void cudaPrepareHeapForCleanup(SimulationData data)
{
    data.tempObjects.heap.reset();
}

__global__ void cudaCleanupCellsStep1(Array<Cell*> cells, Heap newHeap)
{
    // Assumes that cellPointers are already cleaned up
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    int numCellsToCopy = cellPartition.numElements();
    if (numCellsToCopy > 0) {
        auto newCells = newHeap.getTypedSubArray<Cell>(numCellsToCopy);
        auto newHeapStart = newHeap.getArray();

        int newCellIndex = 0;
        for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
            auto& cell = cells.at(index);
            auto newCell = &newCells[newCellIndex];
            *newCell = *cell;

            cell->tempValue = reinterpret_cast<uint8_t*>(newCell) - newHeapStart;  // Save index of new cell in old cell
            cell = newCell;

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
                connectedCell = reinterpret_cast<Cell*>(newHeapStart + connectedCell->tempValue);
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

__global__ void cudaCleanupDependentCellData(Array<Cell*> cells, Heap newHeap)
{
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
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
    data.objects.particles.swapContent(data.tempObjects.particles);
    data.objects.cells.swapContent(data.tempObjects.cells);
}

__global__ void cudaSwapHeaps(SimulationData data)
{
    data.objects.heap.swapContent(data.tempObjects.heap);
}


__global__ void cudaCleanupParticles(Array<Particle*> particlePointers, Heap newHeap)
{
    // Assumes that particlePointers are already cleaned up
    auto partition = calcAllThreadsPartition(particlePointers.getNumEntries());

    int numParticlesToCopy = partition.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = newHeap.getTypedSubArray<Particle>(numParticlesToCopy);

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

__global__ void cudaCleanupGenomesStep1(Array<Cell*> cells)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->genome) {
            cell->genome->genomeIndex = Genome::GenomeIndex_NotSet;
        }
    }
}

__global__ void cudaCleanupGenomesStep2(Array<Cell*> cells, Heap newHeap)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);
        
        if (cell->genome) {
            auto origGenomeIndex = atomicExch(&cell->genome->genomeIndex, 0);  // 0 = member is currently initialized
            if (origGenomeIndex == Genome::GenomeIndex_NotSet) {
                auto newGenome = newHeap.getTypedSubArray<Genome>(1);
                *newGenome = *cell->genome;

                auto const& genome = cell->genome;
                auto newGenes = newHeap.getTypedSubArray<Gene>(genome->numGenes);
                newGenome->genes = newGenes;

                for (int i = 0, j = genome->numGenes; i < j; ++i) {
                    auto const& gene = &genome->genes[i];
                    auto newGene = &newGenes[i];
                    *newGene = *gene;

                    auto newNodes = newHeap.getTypedSubArray<Node>(gene->numNodes);
                    newGene->nodes = newNodes;

                    for (int i = 0, j = gene->numNodes; i < j; ++i) {
                        auto const& node = &gene->nodes[i];
                        auto newNode = &newNodes[i];
                        *newNode = *node;
                    }
                }
                auto newGenomeIndex = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(newGenome) - newHeap.getArray());
                atomicExch(&cell->genome->genomeIndex, newGenomeIndex);
            } else if (origGenomeIndex != 0) {
                atomicExch(&cell->genome->genomeIndex, origGenomeIndex);
            }
        }
    }
}

__global__ void cudaCleanupGenomesStep3(Array<Cell*> cells, Heap newHeap)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->genome) {
            cell->genome = &newHeap.atType<Genome>(cell->genome->genomeIndex);
        }
    }
}

__global__ void cudaCheckIfCleanupIsNecessary(SimulationData data, bool* result)
{
    if (data.objects.heap.getNumEntries() > data.objects.heap.getCapacity() * Const::ArrayFillPercentage) {
        *result = true;
    } else {
        *result = false;
    }
}
