#include "EditKernels.cuh"

__global__ void cudaColorSelectedObjects(SimulationData data, unsigned char color, bool includeClusters)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if ((0 != object->selected && includeClusters) || (1 == object->selected && !includeClusters)) {
            object->color = color;
        }
    }

    auto const energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (0 != particle->selected) {
            particle->color = color;
        }
    }
}

//assumes that *changeTO.numObjects == 1
__global__ void cudaChangeObject(SimulationData data, TOs changeTO)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        auto const& objectTO = changeTO.objects[0];
        if (object->id == objectTO.id) {
            EntityFactory entityFactory;
            entityFactory.init(&data);
            entityFactory.changeObjectFromTO(changeTO, objectTO, object);
            if (objectTO.type == ObjectType_Cell) {
                auto const& creatureTO = changeTO.creatures[objectTO.typeData.cell.creatureIndex];
                auto const& creature = object->typeData.cell.creature;
                creature->lineageId = creatureTO.lineageId;
                creature->prevLineageId = creatureTO.prevLineageId;
                creature->accumulatedMutations = creatureTO.accumulatedMutations;
            }
        }
    }
}

//assumes that *changeTO.numEnergyParticles == 1
__global__ void cudaChangeParticle(SimulationData data, TOs changeTO)
{
    auto const partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& particle = data.entities.energies.at(index);
        auto const& particleTO = changeTO.energyParticles[0];
        if (particle->id == particleTO.id) {
            EntityFactory entityFactory;
            entityFactory.init(&data);
            entityFactory.changeEnergyFromTO(particleTO, particle);
        }
    }
}

__global__ void cudaCreateGenomeFromTO(SimulationData data, TOs to, Genome** newGenome)
{
    EntityFactory factory;
    factory.init(&data);
    *newGenome = factory.createGenomeFromTO(to, 0);
}

__global__ void cudaInjectGenomeToSelectedCreatures(SimulationData data, Genome** newGenome, int* counter)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (object->selected != 1) {
            continue;
        }
        auto oldGenome = alienAtomicExch(&object->typeData.cell.creature->genome, *newGenome);
        if (oldGenome != *newGenome) {
            atomicAdd(counter, 1);
        }
    }
}

namespace
{
    __inline__ __device__ bool isSelected(Object* object, bool includeClusters)
    {
        return (includeClusters && object->selected != 0) || (!includeClusters && object->selected == 1);
    }
}

__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters)
{
    {
        auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& object = data.entities.objects.at(index);
            if (isSelected(object, includeClusters)) {
                object = nullptr;
            }
        }
    }
    {
        auto const partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto& particle = data.entities.energies.at(index);
            if (particle->selected == 1) {
                particle = nullptr;
            }
        }
    }
}

__global__ void cudaRemoveSelectedObjectConnections(SimulationData data, bool includeClusters)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = data.entities.objects.at(index);
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            if ((includeClusters && object->selected != 0) || (!includeClusters && (object->selected == 1 || connectedObject->selected == 1))) {
                ObjectConnectionProcessor::deleteConnectionOneWay(object, connectedObject);
                --i;
            }
        }
    }
}

__global__ void cudaRelaxSelectedEntities(SimulationData data, bool includeClusters)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            auto const numConnections = object->numConnections;
            for (int i = 0; i < numConnections; ++i) {
                auto connectedObject = object->connections[i].object;
                if (isSelected(connectedObject, includeClusters)) {
                    auto delta = connectedObject->pos - object->pos;
                    data.objectMap.correctDirection(delta);
                    object->connections[i].distance = Math::length(delta);
                }
            }

            if (numConnections > 1) {
                for (int i = 0; i < numConnections; ++i) {
                    auto prevConnectedObject = object->connections[(i + numConnections - 1) % numConnections].object;
                    auto connectedObject = object->connections[i].object;
                    if (isSelected(connectedObject, includeClusters) && isSelected(prevConnectedObject, includeClusters)) {
                        auto prevDisplacement = prevConnectedObject->pos - object->pos;
                        data.objectMap.correctDirection(prevDisplacement);
                        auto prevAngle = Math::angleOfVector(prevDisplacement);

                        auto displacement = connectedObject->pos - object->pos;
                        data.objectMap.correctDirection(displacement);
                        auto angle = Math::angleOfVector(displacement);

                        auto actualAngleFromPrevious = Math::subtractAngle(angle, prevAngle);
                        auto angleDiff = actualAngleFromPrevious - object->connections[i].angleFromPrevious;

                        auto nextAngleFromPrevious = object->connections[(i + 1) % numConnections].angleFromPrevious;
                        if (nextAngleFromPrevious - angleDiff >= 0) {
                            object->connections[i].angleFromPrevious = actualAngleFromPrevious;
                            object->connections[(i + 1) % numConnections].angleFromPrevious = nextAngleFromPrevious - angleDiff;
                        }
                    }
                }
            }
        }
    }
}

__global__ void cudaScheduleConnectSelection(SimulationData data, bool considerWithinSelection, int* result)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = data.entities.objects.at(index);
        if (1 != object->selected) {
            continue;
        }
        data.objectMap.executeForEach(object->pos, 1.3f, object->detached(), [&](auto const& otherObject) {
            if (!otherObject || otherObject == object) {
                return;
            }
            if (1 == otherObject->selected && !considerWithinSelection) {
                return;
            }

            auto posDelta = object->pos - otherObject->pos;
            data.objectMap.correctDirection(posDelta);

            for (int i = 0; i < object->numConnections; ++i) {
                auto const& connectedObject = object->connections[i].object;
                if (connectedObject == otherObject) {
                    return;
                }
            }
            if (object->numConnections < MAX_OBJECT_CONNECTIONS && otherObject->numConnections < MAX_OBJECT_CONNECTIONS) {
                ObjectConnectionProcessor::scheduleAddConnectionPair(data, object, otherObject);
                atomicExch(result, 1);
            }
        });
    }
}

__global__ void cudaPrepareMapForReconnection(SimulationData data)
{
    ObjectProcessor::init(data);
}

__global__ void cudaUpdateMapForReconnection(SimulationData data)
{
    ObjectProcessor::updateMap(data);
}

__global__ void cudaUpdateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center)
{
    __shared__ Math::Matrix rotationMatrix;
    if (0 == threadIdx.x) {
        Math::rotationMatrix(updateData.angleDelta, rotationMatrix);
    }
    __syncthreads();

    {
        auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& object = data.entities.objects.at(index);
            if ((updateData.considerClusters && object->selected != 0) || (!updateData.considerClusters && object->selected == 1)) {
                auto relPos = object->pos - center;
                data.objectMap.correctDirection(relPos);

                if (updateData.angleDelta != 0) {
                    object->pos = Math::applyMatrix(relPos, rotationMatrix) + center;
                    data.objectMap.correctPosition(object->pos);
                }

                if (updateData.angularVel != 0) {
                    auto newVel = relPos;
                    Math::rotateQuarterClockwise(newVel);
                    newVel = newVel * updateData.angularVel * Const::DEG_TO_RAD;
                    object->vel = newVel;
                }
            }
        }
    }

    {
        auto const partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = data.entities.energies.at(index);
            if (particle->selected != 0) {
                auto relPos = particle->pos - center;
                data.objectMap.correctDirection(relPos);
                particle->pos = Math::applyMatrix(relPos, rotationMatrix) + center;
                data.objectMap.correctPosition(particle->pos);
            }
        }
    }
}

__global__ void
cudaCalcAccumulatedCenterAndVel(SimulationData data, int refObjectIndex, float2* center, float2* velocity, int* numEntities, bool includeClusters)
{
    {
        float2 refPos = refObjectIndex != -1 ? data.entities.objects.at(refObjectIndex)->pos : float2{0, 0};

        auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& object = data.entities.objects.at(index);
            if (isSelected(object, includeClusters)) {
                if (center) {
                    auto pos = object->pos + data.objectMap.getCorrectionIncrement(refPos, object->pos);
                    atomicAdd(&center->x, pos.x);
                    atomicAdd(&center->y, pos.y);
                }
                if (velocity) {
                    atomicAdd(&velocity->x, object->vel.x);
                    atomicAdd(&velocity->y, object->vel.y);
                }
                atomicAdd(numEntities, 1);
            }
        }
    }
    {
        auto const partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = data.entities.energies.at(index);
            if (particle->selected != 0) {
                if (center) {
                    atomicAdd(&center->x, particle->pos.x);
                    atomicAdd(&center->y, particle->pos.y);
                }
                if (velocity) {
                    atomicAdd(&velocity->x, particle->vel.x);
                    atomicAdd(&velocity->y, particle->vel.y);
                }
                atomicAdd(numEntities, 1);
            }
        }
    }
}

__global__ void cudaIncrementPosAndVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, updateData.considerClusters)) {
            object->pos = object->pos + float2{updateData.posDeltaX, updateData.posDeltaY};
            data.objectMap.correctPosition(object->pos);
            object->vel = float2{updateData.velX, updateData.velY};
        }
    }

    auto const energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (0 != particle->selected) {
            particle->pos = particle->pos + float2{updateData.posDeltaX, updateData.posDeltaY};
            data.energyMap.correctPosition(particle->pos);
            particle->vel = float2{updateData.velX, updateData.velY};
        }
    }
}

__global__ void cudaSetVelocityForSelection(SimulationData data, float2 velocity, bool includeClusters)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->vel = velocity;
        }
    }

    auto const energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (0 != particle->selected) {
            particle->vel = velocity;
        }
    }
}

__global__ void cudaMakeSticky(SimulationData data, bool includeClusters)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->setSticky(true);
        }
    }
}

__global__ void cudaRemoveStickiness(SimulationData data, bool includeClusters)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->setSticky(false);
        }
    }
}

__global__ void cudaSetBarrier(SimulationData data, bool value, bool includeClusters)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->setStatic(value);
        }
    }
}

__global__ void cudaScheduleDisconnectSelectionFromRemainings(SimulationData data, int* result)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (1 == object->selected) {
            for (int i = 0; i < object->numConnections; ++i) {
                auto const& connectedObject = object->connections[i].object;

                if (1 != connectedObject->selected
                    && data.objectMap.getDistance(object->pos, connectedObject->pos) > cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                    ObjectConnectionProcessor::scheduleDeleteConnectionPair(data, object, connectedObject);
                    atomicExch(result, 1);
                }
            }
        }
    }
}

__global__ void cudaPrepareConnectionChanges(SimulationData data)
{
    data.structuralOperations.saveNumEntries();
    ObjectConnectionProcessor::processAddOperations(data);
    ObjectConnectionProcessor::processDeleteConnectionObjectOperations_prepare(data);
}

__global__ void cudaProcessDeleteConnectionChanges(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteConnectionObjectOperations_step1(data);
}

__global__ void cudaProcessAddConnectionChanges(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteConnectionObjectOperations_step2(data);
}

__global__ void cudaApplyForce(SimulationData data, ApplyForceData applyData)
{
    {
        auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& object = data.entities.objects.at(index);
            auto pos = object->pos;
            pos += data.objectMap.getCorrectionIncrement(applyData.startPos, pos);
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius && !object->isStatic()) {
                auto weightedForce = applyData.force;
                //*(actionRadius - distanceToSegment) / actionRadius;
                object->vel = object->vel + weightedForce;
            }
        }
    }
    {
        auto const energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

        for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
            auto const& particle = data.entities.energies.at(index);
            auto const& pos = particle->pos;
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius) {
                auto weightedForce = applyData.force;  //*(actionRadius - distanceToSegment) / actionRadius;
                particle->vel = particle->vel + weightedForce;
            }
        }
    }
}

__global__ void cudaSetDetached(SimulationData data, bool value)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected) {
            object->setDetached(value);
        }
    }
}

__global__ void cudaApplyCataclysm(SimulationData data)
{
    //auto& objects = data.entities.objects;
    //auto partition = calcAllThreadsPartition(cells.getNumEntries());

    //for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
    //    auto& object = cells.at(index);

    //    if (object->typeData.cell.cellType == CellType_Constructor) {
    //        if (data.primaryNumberGen.random() < 0.3f) {
    //            for (int j = 0; j < 100; ++j) {
    //                MutationProcessor::neuronDataMutation(data, object);
    //            }
    //            for (int j = 0; j < 50; ++j) {
    //                MutationProcessor::propertiesMutation(data, object);
    //            }
    //            MutationProcessor::geometryMutation(data, object);
    //            MutationProcessor::customGeometryMutation(data, object);
    //            MutationProcessor::cellTypeMutation(data, object);
    //            int num = data.primaryNumberGen.random(5);
    //            for (int i = 0; i < num; ++i) {
    //                MutationProcessor::insertMutation(data, object);
    //            }
    //            //                MutationProcessor::translateMutation(data, object);
    //            for (int i = 0; i < 2; ++i) {
    //                MutationProcessor::duplicateMutation(data, object);
    //            }
    //            //                MutationProcessor::deleteMutation(data, object);
    //        }
    //    }
    //}
}

__global__ void cudaResetSelectionResult(SelectionResult result)
{
    result.reset();
}

__global__ void cudaCalcObjectWithMinimalPosY(SimulationData data, unsigned long long int* minObjectPosYAndIndex)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected) {
            atomicMin(minObjectPosYAndIndex, (static_cast<unsigned long long int>(abs(object->pos.y)) << 32) | static_cast<unsigned long long int>(index));
        }
    }
}

__global__ void cudaGetSelectionShallowData_step1(SimulationData data)
{
    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected && object->type == ObjectType_Cell) {
            object->typeData.cell.creature->creatureIndex = 0;
        }
    }
}

__global__ void cudaGetSelectionShallowData_step2(SimulationData data, int refObjectIndex, SelectionResult result)
{
    float2 refPos = refObjectIndex != 0xffffffff ? data.entities.objects.at(refObjectIndex)->pos : float2{0, 0};

    auto const objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected) {
            result.collectObject(object, refPos, data.objectMap);
            if (object->selected == 1 && object->type == ObjectType_Cell) {
                if (alienAtomicExch64(&object->typeData.cell.creature->creatureIndex, static_cast<uint64_t>(1)) == static_cast<uint64_t>(0)) {
                    result.collectCreature();
                }
            }
        }
    }

    auto const energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (0 != particle->selected) {
            result.collectParticle(particle, refPos, data.objectMap);
        }
    }
}

__global__ void cudaFinalizeSelectionResult(SelectionResult result, BaseMap map)
{
    result.finalize(map, !cudaSimulationParameters.borderlessRendering.value);
}
