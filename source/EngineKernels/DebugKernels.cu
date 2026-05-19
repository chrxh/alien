#include "DebugKernels.cuh"
#include "ObjectConnectionProcessor.cuh"

namespace
{
    __device__ bool isEnergyValid(float energy)
    {
        return energy >= 0.0f && isfinite(energy);
    }
}

__device__ void DEBUG_checkCells(SimulationData& data, float* sumEnergy, int location)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        if (auto& object = objects.at(index)) {

            if (!isPointerValid(object, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                printf("wrong cell pointer at %d\n", location);
                DEVICE_THROW_NOT_IMPLEMENTED();
            }

            if (object->type == ObjectType_Cell) {
                if (!isPointerValid(object->typeData.cell.creature, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                    printf("wrong creature pointer at %d\n", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
                if (!isPointerValid(object->typeData.cell.creature->genome, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                    printf("wrong genome pointer at %d\n", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
                if (object->typeData.cell.creature->genome->numGenes > 0) {
                    if (!isPointerValid(object->typeData.cell.creature->genome->genes, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                        printf("wrong genes pointer at %d\n", location);
                        DEVICE_THROW_NOT_IMPLEMENTED();
                    }
                }
                for (int i = 0; i < object->typeData.cell.creature->genome->numGenes; ++i) {
                    auto const& gene = object->typeData.cell.creature->genome->genes[i];
                    if (gene.numNodes > 0) {
                        if (!isPointerValid(gene.nodes, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                            printf("wrong nodes pointer at %d\n", location);
                            DEVICE_THROW_NOT_IMPLEMENTED();
                        }
                    }
                }
                if (!isEnergyValid(object->typeData.cell.usableEnergy)) {
                    printf("usable cell energy invalid at %d", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
                if (!isEnergyValid(object->typeData.cell.rawEnergy)) {
                    printf("raw cell energy invalid at %d", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
                if (object->typeData.cell.constructorAvailable && !isEnergyValid(object->typeData.cell.constructor.reservedEnergy)) {
                    printf("reserved cell energy invalid at %d", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
                if (object->typeData.cell.cellType == CellType_Depot && !isEnergyValid(object->typeData.cell.cellTypeData.depot.storedUsableEnergy)) {
                    printf("stored cell energy invalid at %d", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
            }
            if (object->type == ObjectType_FreeCell) {
                if (!isEnergyValid(object->typeData.freeCell.energy)) {
                    printf("raw cell energy invalid at %d", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
            }

            if (object->numConnections > MAX_OBJECT_CONNECTIONS) {
                printf("too much cell connections at %d\n", location);
                DEVICE_THROW_NOT_IMPLEMENTED();
            }
            for (int i = 0; i < object->numConnections; ++i) {
                auto connectedObject = object->connections[i].object;
                if (!isPointerValid(connectedObject, data.entities.heap.getArray(), data.entities.heap.getCapacity())) {
                    printf("wrong connectedObject pointer (cell: %llu, numConnections: %d) at %d\n", object->id, object->numConnections, location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }

                auto displacement = connectedObject->pos - object->pos;
                data.objectMap.correctDirection(displacement);
                auto actualDistance = Math::length(displacement);
                if (actualDistance > 14) {
                    printf("distance too large at %d\n", location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }

                auto connectionOnOtherSideFound = false;
                for (int j = 0; j < connectedObject->numConnections; ++j) {
                    auto connectedConnectedObject = connectedObject->connections[j].object;
                    if (object == connectedConnectedObject) {
                        connectionOnOtherSideFound = true;
                        break;
                    }
                }
                if (!connectionOnOtherSideFound) {
                    printf("connection not found on other side, id=%llu, otherId=%llu,  %d\n", object->id, connectedObject->id, location);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
            }
            if (sumEnergy != nullptr) {
                atomicAdd(sumEnergy, object->getEnergy());
            }
        }
    }
}

__device__ void DEBUG_checkParticles(SimulationData& data, float* sumEnergy, int location)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        if (auto& particle = data.entities.energies.at(particleIndex)) {
            if (reinterpret_cast<uint64_t>(particle) < reinterpret_cast<uint64_t>(data.entities.heap.getArray())
                || reinterpret_cast<uint64_t>(particle) + sizeof(Energy)
                    >= reinterpret_cast<uint64_t>(data.entities.heap.getArray() + data.entities.heap.getCapacity())) {
                printf("wrong particle pointer at %d\n", location);
                DEVICE_THROW_NOT_IMPLEMENTED();
            }
            if (particle->energy < 0 || isnan(particle->energy)) {
                printf("particle energy invalid at %d", location);
                DEVICE_THROW_NOT_IMPLEMENTED();
            }
            if (sumEnergy != nullptr) {
                atomicAdd(sumEnergy, particle->energy);
            }
        }
    }
}

__global__ void DEBUG_checkAngles(SimulationData data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        if (auto& object = objects.at(index)) {
            if (object->numConnections > 0) {
                float sumAngles = 0;
                for (int i = 0; i < object->numConnections; ++i) {
                    sumAngles += object->connections[i].angleFromPrevious;
                    if (object->connections[i].angleFromPrevious < -NEAR_ZERO) {
                        printf("invalid angle: %f\n", object->connections[i].angleFromPrevious);
                        DEVICE_THROW_NOT_IMPLEMENTED();
                    }
                    if (object->connections[i].angleFromPrevious < NEAR_ZERO) {
                        printf("zero angle\n");
                    }
                }
                if (abs(360.0f - sumAngles) > 0.1f) {
                    printf("invalid angle sum: %f\n", sumAngles);
                    DEVICE_THROW_NOT_IMPLEMENTED();
                }
            }
        }
    }
}

__global__ void DEBUG_checkCellsAndParticles(SimulationData data, float* sumEnergy, int location)
{
    DEBUG_checkCells(data, sumEnergy, location);
    DEBUG_checkParticles(data, sumEnergy, location);
}

//__global__ void DEBUG_kernel(SimulationData data, int location)
//{
//    float* sumEnergy = new float;
//    *sumEnergy = 0;
//
//    DEPRECATED_KERNEL_CALL_SYNC(DEBUG_checkCellsAndParticles, data, sumEnergy, location);
//
//    float const expectedEnergy = 187500;
//    if (abs(*sumEnergy - expectedEnergy) > 1) {
//        printf("location: %d, actual energy: %f, expected energy: %f\n", location, *sumEnergy, expectedEnergy);
//        CUDA_THROW_NOT_IMPLEMENTED();
//    }
//    delete sumEnergy;
//}
