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

            cell->tempValue.as_uint64 = reinterpret_cast<uint8_t*>(newCell) - newHeapStart;  // Save index of new cell in old cell
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
                connectedCell = reinterpret_cast<Cell*>(newHeapStart + connectedCell->tempValue.as_uint64);
            }
            if (cell->cellType == CellType_Constructor) {
                cell->cellTypeData.constructor.offspring = nullptr;
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
        if (cell->neuralNetwork) {
            copyAndAssignNewHeapData(reinterpret_cast<uint8_t*&>(cell->neuralNetwork), sizeof(*cell->neuralNetwork), newHeap);
        }
        if (cell->cellType == CellType_Memory) {
            copyAndAssignNewHeapData(
                reinterpret_cast<uint8_t*&>(cell->cellTypeData.memory.memoryEntries),
                sizeof(MemoryEntry) * MAX_CELL_MEMORY_ENTRIES,
                newHeap);
        }
    }
}

__global__ void cudaCleanupMaps(SimulationData data)
{
    data.cellMap.cleanup_system();
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

__global__ void cudaPrepareCleanupCreaturesAndGenomes(Array<Cell*> cells)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->creature) {
            cell->creature->creatureIndex = VALUE_NOT_SET_UINT64;
            cell->creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaCleanupGenomesStep1(Array<Cell*> cells, Heap newHeap)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->creature) {
            auto const& genome = cell->creature->genome;
            auto origGenomeIndex = alienAtomicExch64(&genome->genomeIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized

            if (origGenomeIndex == VALUE_NOT_SET_UINT64) {
                auto newGenome = newHeap.getTypedSubArray<Genome>(1);
                *newGenome = *genome;

                auto newGenes = newHeap.getTypedSubArray<Gene>(genome->numGenes);
                newGenome->genes = newGenes;

                for (int i = 0, j = genome->numGenes; i < j; ++i) {
                    auto const& gene = &genome->genes[i];
                    auto newGene = &newGenes[i];
                    *newGene = *gene;

                    auto newNodes = newHeap.getTypedSubArray<Node>(gene->numNodes);
                    newGene->nodes = newNodes;

                    for (int k = 0, l = gene->numNodes; k < l; ++k) {
                        auto const& node = &gene->nodes[k];
                        auto newNode = &newNodes[k];
                        *newNode = *node;

                        // Copy dependent node data for memory cell type
                        if (node->cellType == CellTypeGenome_Memory) {
                            copyAndAssignNewHeapData(
                                reinterpret_cast<uint8_t*&>(newNode->cellTypeData.memory.memoryEntries),
                                sizeof(MemoryEntryGenome) * MAX_CELL_MEMORY_ENTRIES,
                                newHeap);
                        }
                    }
                }

                auto newGenomeIndex = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(newGenome) - newHeap.getArray());
                alienAtomicExch64(&genome->genomeIndex, newGenomeIndex);
                newGenome->genomeIndex = newGenomeIndex;

            } else if (origGenomeIndex != 0) {
                alienAtomicExch64(&genome->genomeIndex, origGenomeIndex);
            }
        }
    }
}

__global__ void cudacudaCleanupGenomesStep2(Array<Cell*> cells, Heap newHeap)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->creature) {
            cell->creature->genome = &newHeap.atType<Genome>(cell->creature->genome->genomeIndex);
        }
    }
}

__global__ void cudaCleanupCreaturesStep1(Array<Cell*> cells, Heap newHeap)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->creature) {
            auto origCreatureIndex = alienAtomicExch64(&cell->creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
            if (origCreatureIndex == VALUE_NOT_SET_UINT64) {
                auto newCreature = newHeap.getTypedSubArray<Creature>(1);
                auto const& creature = cell->creature;
                *newCreature = *creature;

                auto newCreatureIndex = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(newCreature) - newHeap.getArray());
                alienAtomicExch64(&cell->creature->creatureIndex, newCreatureIndex);
            } else if (origCreatureIndex != 0) {
                alienAtomicExch64(&cell->creature->creatureIndex, origCreatureIndex);
            }
        }
    }
}

__global__ void cudaCleanupCreaturesStep2(Array<Cell*> cells, Heap newHeap)
{
    PartitionData cellPartition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto& cell = cells.at(index);
        auto const& creature = cell->creature;
        if (creature) {
            cell->creature = &newHeap.atType<Creature>(creature->creatureIndex);
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
