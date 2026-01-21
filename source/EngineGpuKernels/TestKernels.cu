#include "ObjectConnectionProcessor.cuh"
#include "SimulationData.cuh"
#include "TestKernels.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t objectId, MutationType mutationType)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        //if (object->id == objectId) {
        //    switch (mutationType) {
        //    case MutationType::Properties:
        //        MutationProcessor::propertiesMutation(data, object);
        //        break;
        //    case MutationType::NeuronData:
        //        MutationProcessor::neuronDataMutation(data, object);
        //        break;
        //    case MutationType::Geometry:
        //        MutationProcessor::geometryMutation(data, object);
        //        break;
        //    case MutationType::CustomGeometry:
        //        MutationProcessor::customGeometryMutation(data, object);
        //        break;
        //    case MutationType::CellType:
        //        MutationProcessor::cellTypeMutation(data, object);
        //        break;
        //    case MutationType::Insertion:
        //        MutationProcessor::insertMutation(data, object);
        //        break;
        //    case MutationType::Deletion:
        //        MutationProcessor::deleteMutation(data, object);
        //        break;
        //    case MutationType::Translation:
        //        MutationProcessor::translateMutation(data, object);
        //        break;
        //    case MutationType::Duplication:
        //        MutationProcessor::duplicateMutation(data, object);
        //        break;
        //    case MutationType::CellColor:
        //        MutationProcessor::cellColorMutation(data, object);
        //        break;
        //    case MutationType::SubgenomeColor:
        //        MutationProcessor::subgenomeColorMutation(data, object);
        //        break;
        //    case MutationType::GenomeColor:
        //        MutationProcessor::genomeColorMutation(data, object);
        //        break;
        //    }
        //}
    }
}

__global__ void cudaTestCreateConnection(SimulationData data, uint64_t objectId1, uint64_t objectId2)
{
    CUDA_CHECK(blockDim.x == 1 && gridDim.x == 1);

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

namespace
{
    template<typename Pointer>
    __device__ bool isPointerValid(SimulationData const& data, Pointer pointer)
    {
        return reinterpret_cast<uint64_t>(pointer) >= reinterpret_cast<uint64_t>(data.entities.heap.getArray())
            && reinterpret_cast<uint64_t>(pointer) < reinterpret_cast<uint64_t>(data.entities.heap.getArray() + data.entities.heap.getCapacity());
    }

    __device__ bool isGenomeValid(SimulationData const& data, Genome* genome)
    {
        if (genome != nullptr) {
            if (!isPointerValid(data, genome)) {
               return false;
            } else {

                // Validate genes pointer
                if (genome->numGenes > 0) {
                    if (!isPointerValid(data, genome->genes)) {
                        return false;
                    } else {
                        // Validate each gene's nodes pointer and memory node entries
                        for (int geneIdx = 0; geneIdx < genome->numGenes; ++geneIdx) {
                            auto& gene = genome->genes[geneIdx];
                            if (gene.numNodes > 0) {
                                if (!isPointerValid(data, gene.nodes)) {
                                    return false;
                                } else {
                                    // Validate memory entries for memory nodes
                                    for (int nodeIdx = 0; nodeIdx < gene.numNodes; ++nodeIdx) {
                                        auto& node = gene.nodes[nodeIdx];
                                        if (node.cellType == CellType_Memory) {
                                            if (node.cellTypeData.memory.numSignalEntries > 0) {
                                                auto signalEntries = node.cellTypeData.memory.signalEntries;
                                                if (!isPointerValid(data, signalEntries)) {
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

__global__ void cudaTestArePointersValid(SimulationData data, bool* result)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        if (auto& object = objects.at(index)) {

            if (isPointerValid(data, object)) {
                for (int i = 0; i < object->numConnections; ++i) {
                    auto connectedObject = object->connections[i].object;
                    *result &= isPointerValid(data, connectedObject);
                }

                if (object->type == ObjectType_Cell) {
                    if (object->typeData.cell.cellType == CellType_Memory) {
                        if (object->typeData.cell.cellTypeData.memory.numSignalEntries > 0) {
                            auto signalEntries = object->typeData.cell.cellTypeData.memory.signalEntries;
                            *result &= isPointerValid(data, signalEntries);
                        }
                    }

                    if (object->typeData.cell.neuralNetwork != nullptr) {
                        *result &= isPointerValid(data, object->typeData.cell.neuralNetwork);
                    }

                    if (object->typeData.cell.creature != nullptr) {
                        if (!isPointerValid(data, object->typeData.cell.creature)) {
                            *result = false;
                        } else {
                            *result &= isGenomeValid(data, object->typeData.cell.creature->genome);
                        }
                    }
                }
            } else {
                *result = false;
            }
        }
    }
}

__global__ void cudaTestMutationCheck(SimulationData data, uint64_t objectId)
{
    //auto& objects = data.entities.objects;
    //auto partition = calcAllThreadsPartition(cells.getNumEntries());

    //for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
    //    auto& object = cells.at(index);
    //    if (object->id == objectId) {
    //        MutationProcessor::checkMutationsForCell(data, object);
    //    }
    //}
}
