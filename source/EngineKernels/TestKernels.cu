#include <cooperative_groups.h>

#include "MutationProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "TestKernels.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t objectId)
{
    DEVICE_CHECK(blockDim.x == NEURONS_PER_CELL);

    auto block = cooperative_groups::this_thread_block();
    auto laneId = block.thread_rank();

    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& object = objects.at(index);

        __shared__ bool shouldMutate;
        if (laneId == 0) {
            DEVICE_CHECK(object->type == ObjectType_Cell);
            shouldMutate = (object->id == objectId);
        }
        block.sync();

        if (shouldMutate) {
            MutationProcessor::applyMutations(data, object->typeData.cell.creature->genome);
        }
        block.sync();
    }
}

__global__ void cudaTestCreateConnection(SimulationData data, uint64_t objectId1, uint64_t objectId2)
{
    DEVICE_CHECK(blockDim.x == 1 && gridDim.x == 1);

    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    Object* object1 = nullptr;
    Object* object2 = nullptr;
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->id == objectId1) {
            object1 = object;
        }
        if (object->id == objectId2) {
            object2 = object;
        }
    }

    if (object1 != nullptr && object2 != nullptr) {
        data.objectMap.reset();
        data.energyMap.reset();
        data.processMemory.reset();

        // Heuristics
        auto maxStructureOperations = 1000 + data.entities.objects.getNumEntries() / 2;
        auto maxCellTypeOperations = data.entities.objects.getNumEntries();

        data.structuralOperations.setMemory(data.processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);
        for (int i = CellType_Base; i < CellType_Count; ++i) {
            data.cellTypeOperations[i].setMemory(data.processMemory.getTypedSubArray<CellTypeOperation>(maxCellTypeOperations), maxCellTypeOperations);
        }
        *data.externalEnergy = cudaSimulationParameters.externalEnergy.value;
        data.entities.saveNumEntries();

        ObjectConnectionProcessor::scheduleAddConnectionPair(data, object1, object2);
        data.structuralOperations.saveNumEntries();
        ObjectConnectionProcessor::processAddOperations(data);
    }
}

__global__ void cudaTestCreateConnectionWithAbsAngle(
    SimulationData data,
    uint64_t objectId1,
    uint64_t objectId2,
    float desiredDistance,
    float desiredAbsAngle1,
    float desiredAbsAngle2)
{
    DEVICE_CHECK(blockDim.x == 1 && gridDim.x == 1);

    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    Object* object1 = nullptr;
    Object* object2 = nullptr;
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->id == objectId1) {
            object1 = object;
        }
        if (object->id == objectId2) {
            object2 = object;
        }
    }

    if (object1 != nullptr && object2 != nullptr) {
        ObjectConnectionProcessor::tryAddConnectionWithAbsAngle(data, object1, object2, desiredDistance, desiredAbsAngle1, desiredAbsAngle2);
    }
}

namespace
{
    __device__ bool isEnergyValid(float energy)
    {
        return energy >= 0.0f && isfinite(energy);
    }

    __device__ bool isGenomeValid(SimulationData const& data, Genome* genome)
    {
        if (genome != nullptr) {
            if (!isPointerValid(genome, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                return false;
            } else {

                // Validate genes pointer
                if (genome->numGenes > 0) {
                    if (!isPointerValid(genome->genes, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                        return false;
                    } else {
                        // Validate each gene's nodes pointer and memory node entries
                        for (int geneIdx = 0; geneIdx < genome->numGenes; ++geneIdx) {
                            auto& gene = genome->genes[geneIdx];
                            if (gene.numNodes > 0) {
                                if (!isPointerValid(gene.nodes, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                                    return false;
                                } else {
                                    // Validate memory entries for memory nodes
                                    for (int nodeIdx = 0; nodeIdx < gene.numNodes; ++nodeIdx) {
                                        auto& node = gene.nodes[nodeIdx];
                                        if (node.cellType == CellType_Memory) {
                                            if (node.cellTypeData.memory.numSignalEntries > 0) {
                                                auto signalEntries = node.cellTypeData.memory.signalEntries;
                                                if (!isPointerValid(signalEntries, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                                                    return false;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
}

__global__ void cudaTestIsDataValid(SimulationData data, bool* result)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        if (auto& object = objects.at(index)) {

            if (isPointerValid(object, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                if (object->numConnections > MAX_OBJECT_CONNECTIONS) {
                    *result = false;
                    continue;
                }

                if (object->type == ObjectType_Cell) {
                    *result &= isEnergyValid(object->typeData.cell.usableEnergy);
                    *result &= isEnergyValid(object->typeData.cell.rawEnergy);
                    if (object->typeData.cell.constructorAvailable) {
                        *result &= isEnergyValid(object->typeData.cell.constructor.reservedEnergy);
                    }
                    if (object->typeData.cell.cellType == CellType_Depot) {
                        *result &= isEnergyValid(object->typeData.cell.cellTypeData.depot.storedUsableEnergy);
                    }
                    if (object->typeData.cell.cellType == CellType_Memory) {
                        if (object->typeData.cell.cellTypeData.memory.numSignalEntries > 0) {
                            auto signalEntries = object->typeData.cell.cellTypeData.memory.signalEntries;
                            *result &= isPointerValid(signalEntries, data.entities.heap.getArray(), data.entities.heap.getCapacity());
                        }
                    }

                    if (object->typeData.cell.neuralNetwork != nullptr) {
                        *result &= isPointerValid(object->typeData.cell.neuralNetwork, data.entities.heap.getArray(), data.entities.heap.getCapacity());
                    }

                    if (object->typeData.cell.creature != nullptr) {
                        if (!isPointerValid(object->typeData.cell.creature, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                            *result = false;
                        } else {
                            *result &= isGenomeValid(data, object->typeData.cell.creature->genome);
                        }
                    }
                }
                if (object->type == ObjectType_FreeCell) {
                    *result &= isEnergyValid(object->typeData.freeCell.energy);
                }

                for (int i = 0; i < object->numConnections; ++i) {
                    auto connectedObject = object->connections[i].object;
                    auto connectedObjectValid = isPointerValid(connectedObject, data.entities.heap.getArray(), data.entities.heap.getCapacity());
                    *result &= connectedObjectValid;
                    if (!connectedObjectValid) {
                        continue;
                    }
                    if (connectedObject->numConnections > MAX_OBJECT_CONNECTIONS) {
                        *result = false;
                        continue;
                    }

                    auto displacement = connectedObject->pos - object->pos;
                    data.objectMap.correctDirection(displacement);
                    auto actualDistance = Math::length(displacement);
                    *result &= actualDistance <= 14.0f;

                    auto connectionOnOtherSideFound = false;
                    for (int j = 0; j < connectedObject->numConnections; ++j) {
                        auto connectedConnectedObject = connectedObject->connections[j].object;
                        if (object == connectedConnectedObject) {
                            connectionOnOtherSideFound = true;
                            break;
                        }
                    }
                    *result &= connectionOnOtherSideFound;
                }
            } else {
                *result = false;
            }
        }
    }
}
