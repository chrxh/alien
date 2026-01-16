#include "EditKernels.cuh"

__global__ void cudaColorSelectedCells(SimulationData data, unsigned char color, bool includeClusters)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if ((0 != object->selected && includeClusters) || (1 == object->selected && !includeClusters)) {
            object->color = color;
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (0 != particle->selected) {
            particle->color = color;
        }
    }
}

//assumes that *changeTO.numObjects == 1
__global__ void cudaChangeCell(SimulationData data, TO changeTO)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        auto const& cellTO = changeTO.objects[0];
        if (object->id == cellTO.id) {
            EntityFactory entityFactory;
            entityFactory.init(&data);
            entityFactory.changeObjectFromTO(changeTO, cellTO, object);
        }
    }
}

//assumes that *changeTO.numEnergyParticles == 1
__global__ void cudaChangeParticle(SimulationData data, TO changeTO)
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

__global__ void cudaAddGenomeAndCreature(SimulationData data, TO to, Genome** newGenome, Creature** newCreature)
{
    EntityFactory factory;
    factory.init(&data);
    *newGenome = factory.createGenomeFromTO(to, 0);
    *newCreature = factory.createCreatureFromTO(to, 0);
}

__global__ void cudaChangeCellToCreature(SimulationData data, Creature** newCreature, bool* result)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (object->creature->id == (*newCreature)->id) {
            object->creature = *newCreature;
            *result = true;
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
            auto connectedCell = object->connections[i].object;
            if ((includeClusters && object->selected != 0) || (!includeClusters && (object->selected == 1 || connectedCell->selected == 1))) {
                ObjectConnectionProcessor::deleteConnectionOneWay(object, connectedCell);
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
                auto connectedCell = object->connections[i].object;
                if (isSelected(connectedCell, includeClusters)) {
                    auto delta = connectedCell->pos - object->pos;
                    data.objectMap.correctDirection(delta);
                    object->connections[i].distance = Math::length(delta);
                }
            }

            if (numConnections > 1) {
                for (int i = 0; i < numConnections; ++i) {
                    auto prevConnectedCell = object->connections[(i + numConnections - 1) % numConnections].object;
                    auto connectedCell = object->connections[i].object;
                    if (isSelected(connectedCell, includeClusters) && isSelected(prevConnectedCell, includeClusters)) {
                        auto prevDisplacement = prevConnectedCell->pos - object->pos;
                        data.objectMap.correctDirection(prevDisplacement);
                        auto prevAngle = Math::angleOfVector(prevDisplacement);

                        auto displacement = connectedCell->pos - object->pos;
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
        data.objectMap.executeForEach(object->pos, 1.3f, object->detached, [&](auto const& otherCell) {
            if (!otherCell || otherCell == object) {
                return;
            }
            if (1 == otherCell->selected && !considerWithinSelection) {
                return;
            }

            auto posDelta = object->pos - otherCell->pos;
            data.objectMap.correctDirection(posDelta);

            for (int i = 0; i < object->numConnections; ++i) {
                auto const& connectedCell = object->connections[i].object;
                if (connectedCell == otherCell) {
                    return;
                }
            }
            if (object->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS) {
                ObjectConnectionProcessor::scheduleAddConnectionPair(data, object, otherCell);
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

__global__ void cudaCalcAccumulatedCenterAndVel(SimulationData data, int refCellIndex, float2* center, float2* velocity, int* numEntities, bool includeClusters)
{
    {
        float2 refPos = refCellIndex != -1 ? data.entities.objects.at(refCellIndex)->pos : float2{0, 0};

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
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, updateData.considerClusters)) {
            object->pos = object->pos + float2{updateData.posDeltaX, updateData.posDeltaY};
            data.objectMap.correctPosition(object->pos);
            object->vel = float2{updateData.velX, updateData.velY};
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
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
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->vel = velocity;
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.entities.energies.at(index);
        if (0 != particle->selected) {
            particle->vel = velocity;
        }
    }
}

__global__ void cudaMakeSticky(SimulationData data, bool includeClusters)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->sticky = true;
        }
    }
}

__global__ void cudaRemoveStickiness(SimulationData data, bool includeClusters)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->sticky = false;
        }
    }
}

__global__ void cudaSetBarrier(SimulationData data, bool value, bool includeClusters)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (isSelected(object, includeClusters)) {
            object->fixed = value;
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
                auto const& connectedCell = object->connections[i].object;

                if (1 != connectedCell->selected
                    && data.objectMap.getDistance(object->pos, connectedCell->pos) > cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                    ObjectConnectionProcessor::scheduleDeleteConnectionPair(data, object, connectedCell);
                    atomicExch(result, 1);
                }
            }
        }
    }
}

__global__ void cudaPrepareConnectionChanges(SimulationData data)
{
    data.structuralOperations.saveNumEntries();
}

__global__ void cudaProcessDeleteConnectionChanges(SimulationData data)
{
    ObjectConnectionProcessor::processDeleteConnectionOperations(data);
}

__global__ void cudaProcessAddConnectionChanges(SimulationData data)
{
    ObjectConnectionProcessor::processAddOperations(data);
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
            if (distanceToSegment < applyData.radius && !object->fixed) {
                auto weightedForce = applyData.force;
                //*(actionRadius - distanceToSegment) / actionRadius;
                object->vel = object->vel + weightedForce;
            }
        }
    }
    {
        auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

        for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
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
            object->detached = value ? 1 : 0;
        }
    }
}

__global__ void cudaApplyCataclysm(SimulationData data)
{
    //auto& cells = data.entities.objects;
    //auto partition = calcAllThreadsPartition(cells.getNumEntries());

    //for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
    //    auto& object = cells.at(index);

    //    if (object->cellType == CellType_Constructor) {
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

__global__ void cudaCalcCellWithMinimalPosY(SimulationData data, unsigned long long int* minCellPosYAndIndex)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected) {
            atomicMin(minCellPosYAndIndex, (static_cast<unsigned long long int>(abs(object->pos.y)) << 32) | static_cast<unsigned long long int>(index));
        }
    }
}

__global__ void cudaGetSelectionShallowData_step1(SimulationData data)
{
    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected && object->creature != nullptr) {
            object->creature->creatureIndex = 0;
        }
    }
}

__global__ void cudaGetSelectionShallowData_step2(SimulationData data, int refCellIndex, SelectionResult result)
{
    float2 refPos = refCellIndex != 0xffffffff ? data.entities.objects.at(refCellIndex)->pos : float2{0, 0};

    auto const cellPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (0 != object->selected) {
            result.collectCell(object, refPos, data.objectMap);
            if (object->creature != nullptr) {
                if (alienAtomicExch64(&object->creature->creatureIndex, static_cast<uint64_t>(1)) == static_cast<uint64_t>(0)) {
                    result.collectCreature();
                }
            }
        }
    }

    auto const particlePartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());

    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
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
