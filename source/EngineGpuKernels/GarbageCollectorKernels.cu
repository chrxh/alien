#include "GarbageCollectorKernels.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data)
{
    data.tempEntities.energies.reset();
    data.tempEntities.objects.reset();
}

__global__ void cudaPrepareHeapForCleanup(SimulationData data)
{
    data.tempEntities.heap.reset();
}

__global__ void cudaCleanupCellsStep1(Array<Object*> objects, Heap newHeap)
{
    // Assumes that cellPointers are already cleaned up
    auto objectPartition = calcSystemThreadPartition(objects.getNumEntries());

    int numCellsToCopy = objectPartition.numElements();
    if (numCellsToCopy > 0) {
        auto newObjects = newHeap.getTypedSubArray<Object>(numCellsToCopy);
        auto newHeapStart = newHeap.getArray();

        int newObjectIndex = 0;
        for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
            auto& object = objects.at(index);
            auto newObject = &newObjects[newObjectIndex];
            *newObject = *object;

            object->tempValue.as_uint64 = reinterpret_cast<uint8_t*>(newObject) - newHeapStart;  // Save index of new cell in old cell
            object = newObject;

            ++newObjectIndex;
        }
    }
}

__global__ void cudaCleanupCellsStep2(Array<Object*> cellPointers, Heap newHeap)
{
    {
        auto partition = calcSystemThreadPartition(cellPointers.getNumEntries());
        auto newHeapStart = newHeap.getArray();
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = cellPointers.at(index);
            for (int i = 0; i < object->numConnections; ++i) {
                auto& connectedObject = object->connections[i].object;
                connectedObject = reinterpret_cast<Object*>(newHeapStart + connectedObject->tempValue.as_uint64);
            }
            if (object->typeData.cell.cellType == CellType_Constructor) {
                object->typeData.cell.cellTypeData.constructor.offspring = nullptr;
            }
        }
    }
}

namespace
{
    __device__ void copyAndAssignNewHeapData(uint8_t*& data, uint64_t numBytes, Heap& heap)
    {
        if (numBytes > 0) {
            uint8_t* bytes = heap.getRawSubArray(numBytes);
            for (uint64_t i = 0; i < numBytes; ++i) {
                bytes[i] = data[i];
            }
            data = bytes;
        }
    }

    __device__ void copyAndAssignNewHeapData(uint8_t*& target, uint8_t* const& source, uint64_t numBytes, Heap& heap)
    {
        if (numBytes > 0) {
            uint8_t* bytes = heap.getRawSubArray(numBytes);
            for (uint64_t i = 0; i < numBytes; ++i) {
                bytes[i] = source[i];
            }
            target = bytes;
        }
    }
}

__global__ void cudaCleanupDependentCellData(Array<Object*> cells, Heap newHeap)
{
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->type == ObjectType_Cell) {
            copyAndAssignNewHeapData(
                reinterpret_cast<uint8_t*&>(object->typeData.cell.neuralNetwork), sizeof(*object->typeData.cell.neuralNetwork), newHeap);
            if (object->typeData.cell.cellType == CellType_Memory) {
                copyAndAssignNewHeapData(
                    reinterpret_cast<uint8_t*&>(object->typeData.cell.cellTypeData.memory.signalEntries),
                    sizeof(SignalEntry) * object->typeData.cell.cellTypeData.memory.numSignalEntries,
                    newHeap);
            }
        }
    }
}

__global__ void cudaCleanupMaps(SimulationData data)
{
    data.objectMap.cleanup_system();
    data.energyMap.cleanup_system();
}

__global__ void cudaSwapPointerArrays(SimulationData data)
{
    data.entities.energies.swapContent(data.tempEntities.energies);
    data.entities.objects.swapContent(data.tempEntities.objects);
}

__global__ void cudaSwapHeaps(SimulationData data)
{
    data.entities.heap.swapContent(data.tempEntities.heap);
}


__global__ void cudaCleanupParticles(Array<Energy*> particlePointers, Heap newHeap)
{
    // Assumes that particlePointers are already cleaned up
    auto partition = calcSystemThreadPartition(particlePointers.getNumEntries());

    int numParticlesToCopy = partition.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = newHeap.getTypedSubArray<Energy>(numParticlesToCopy);

        int newParticleIndex = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& particlePointer = particlePointers.at(index);
            auto& newParticle = newParticles[newParticleIndex];
            newParticle = *particlePointer;
            particlePointer = &newParticle;

            ++newParticleIndex;
        }
    }
}

__global__ void cudaPrepareCleanupCreaturesAndGenomes(Array<Object*> cells)
{
    auto objectPartition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto& object = cells.at(index);
        if (object->typeData.cell.creature) {
            object->typeData.cell.creature->creatureIndex = VALUE_NOT_SET_UINT64;
            object->typeData.cell.creature->genome->genomeIndex = VALUE_NOT_SET_UINT64;
        }
    }
}

__global__ void cudaCleanupGenomesStep1(Array<Object*> cells, Heap newHeap)
{
    auto objectPartition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto& object = cells.at(index);

        if (object->typeData.cell.creature) {
            auto const& genome = object->typeData.cell.creature->genome;
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
                                reinterpret_cast<uint8_t*&>(newNode->cellTypeData.memory.signalEntries),
                                reinterpret_cast<uint8_t*&>(node->cellTypeData.memory.signalEntries),
                                sizeof(SignalEntryGenome) * newNode->cellTypeData.memory.numSignalEntries,
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

__global__ void cudacudaCleanupGenomesStep2(Array<Object*> cells, Heap newHeap)
{
    auto objectPartition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto& object = cells.at(index);
        if (object->typeData.cell.creature) {
            object->typeData.cell.creature->genome = &newHeap.atType<Genome>(object->typeData.cell.creature->genome->genomeIndex);
        }
    }
}

__global__ void cudaCleanupCreaturesStep1(Array<Object*> cells, Heap newHeap)
{
    auto objectPartition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto& object = cells.at(index);

        if (object->typeData.cell.creature) {
            auto origCreatureIndex = alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, static_cast<uint64_t>(0));  // 0 = member is currently initialized
            if (origCreatureIndex == VALUE_NOT_SET_UINT64) {
                auto newCreature = newHeap.getTypedSubArray<Creature>(1);
                auto const& creature = object->typeData.cell.creature;
                *newCreature = *creature;

                auto newCreatureIndex = static_cast<uint64_t>(reinterpret_cast<uint8_t*>(newCreature) - newHeap.getArray());
                alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, newCreatureIndex);
            } else if (origCreatureIndex != 0) {
                alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, origCreatureIndex);
            }
        }
    }
}

__global__ void cudaCleanupCreaturesStep2(Array<Object*> cells, Heap newHeap)
{
    auto objectPartition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto& object = cells.at(index);
        auto const& creature = object->typeData.cell.creature;
        if (creature) {
            object->typeData.cell.creature = &newHeap.atType<Creature>(creature->creatureIndex);
        }
    }
}

__global__ void cudaCheckIfCleanupIsNecessary(SimulationData data, bool* result)
{
    if (data.entities.heap.getNumEntries() > data.entities.heap.getCapacity() * Const::ArrayFillPercentage) {
        *result = true;
    } else {
        *result = false;
    }
}
