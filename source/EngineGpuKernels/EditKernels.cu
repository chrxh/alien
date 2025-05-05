#include "EditKernels.cuh"

#include "MutationProcessor.cuh"

__global__ void cudaColorSelectedCells(SimulationData data, unsigned char color, bool includeClusters)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if ((0 != cell->selected && includeClusters) || (1 == cell->selected && !includeClusters)) {
            cell->color = color;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->color = color;
        }
    }
}

__global__ void cudaPrepareForUpdate(SimulationData data)
{
    data.prepareForNextTimestep();
}

//assumes that *changeDataTO.numCells == 1
__global__ void cudaChangeCell(SimulationData data, DataTO changeDataTO)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        auto const& cellTO = changeDataTO.cells[0];
        if (cell->id == cellTO.id) {
            ObjectFactory entityFactory;
            entityFactory.init(&data);
            entityFactory.changeCellFromTO(changeDataTO, cellTO, cell, false);
        }
    }
}

//assumes that *changeDataTO.numParticles == 1
__global__ void cudaChangeParticle(SimulationData data, DataTO changeDataTO)
{
    auto const partition = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        auto const& particleTO = changeDataTO.particles[0];
        if (particle->id == particleTO.id) {
            ObjectFactory entityFactory;
            entityFactory.init(&data);
            entityFactory.changeParticleFromTO(particleTO, particle);
        }
    }
}

namespace
{
    __inline__ __device__ bool isSelected(Cell* cell, bool includeClusters)
    {
        return (includeClusters && cell->selected != 0) || (!includeClusters && cell->selected == 1);
    }
}

__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters)
{
    {
        auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = data.objects.cellPointers.at(index);
            if (isSelected(cell, includeClusters)) {
                cell = nullptr;
            }
        }
    }
    {
        auto const partition = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = data.objects.particlePointers.at(index);
            if (particle->selected == 1) {
                particle = nullptr;
            }
        }
    }
}

__global__ void cudaRemoveSelectedCellConnections(SimulationData data, bool includeClusters)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if ((includeClusters && cell->selected != 0) || (!includeClusters && (cell->selected == 1 || connectedCell->selected == 1))) {
                CellConnectionProcessor::deleteConnectionOneWay(cell, connectedCell);
                --i;
            }
        }
    }
}

__global__ void cudaRelaxSelectedEntities(SimulationData data, bool includeClusters)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            auto const numConnections = cell->numConnections;
            for (int i = 0; i < numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                if (isSelected(connectedCell, includeClusters)) {
                    auto delta = connectedCell->pos - cell->pos;
                    data.cellMap.correctDirection(delta);
                    cell->connections[i].distance = Math::length(delta);
                }
            }

            if (numConnections > 1) {
                for (int i = 0; i < numConnections; ++i) {
                    auto prevConnectedCell = cell->connections[(i + numConnections - 1) % numConnections].cell;
                    auto connectedCell = cell->connections[i].cell;
                    if (isSelected(connectedCell, includeClusters) && isSelected(prevConnectedCell, includeClusters)) {
                        auto prevDisplacement = prevConnectedCell->pos - cell->pos;
                        data.cellMap.correctDirection(prevDisplacement);
                        auto prevAngle = Math::angleOfVector(prevDisplacement);

                        auto displacement = connectedCell->pos - cell->pos;
                        data.cellMap.correctDirection(displacement);
                        auto angle = Math::angleOfVector(displacement);

                        auto actualAngleFromPrevious = Math::subtractAngle(angle, prevAngle);
                        auto angleDiff = actualAngleFromPrevious - cell->connections[i].angleFromPrevious;

                        auto nextAngleFromPrevious = cell->connections[(i + 1) % numConnections].angleFromPrevious;
                        if (nextAngleFromPrevious - angleDiff >= 0) {
                            cell->connections[i].angleFromPrevious = actualAngleFromPrevious;
                            cell->connections[(i + 1) % numConnections].angleFromPrevious = nextAngleFromPrevious - angleDiff;
                        }
                    }
                }
            }
        }
    }
}

__global__ void cudaScheduleConnectSelection(SimulationData data, bool considerWithinSelection, int* result)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        if (1 != cell->selected) {
            continue;
        }
        data.cellMap.executeForEach(cell->pos, 1.3f, cell->detached, [&](auto const& otherCell) {
            if (!otherCell || otherCell == cell) {
                return;
            }
            if (1 == otherCell->selected && !considerWithinSelection) {
                return;
            }

            auto posDelta = cell->pos - otherCell->pos;
            data.cellMap.correctDirection(posDelta);

            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (connectedCell == otherCell) {
                    return;
                }
            }
            if (cell->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS) {
                CellConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                atomicExch(result, 1);
            }
        });
    }
}

__global__ void cudaPrepareMapForReconnection(SimulationData data)
{
    CellProcessor::init(data);
}

__global__ void cudaUpdateMapForReconnection(SimulationData data)
{
    CellProcessor::updateMap(data);
}

__global__ void cudaUpdateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center)
{
    __shared__ Math::Matrix rotationMatrix;
    if (0 == threadIdx.x) {
        Math::rotationMatrix(updateData.angleDelta, rotationMatrix);
    }
    __syncthreads();

    {
        auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.objects.cellPointers.at(index);
            if ((updateData.considerClusters && cell->selected != 0) || (!updateData.considerClusters && cell->selected == 1)) {
                auto relPos = cell->pos - center;
                data.cellMap.correctDirection(relPos);

                if (updateData.angleDelta != 0) {
                    cell->pos = Math::applyMatrix(relPos, rotationMatrix) + center;
                    data.cellMap.correctPosition(cell->pos);
                }

                if (updateData.angularVel != 0) {
                    auto newVel = relPos;
                    Math::rotateQuarterClockwise(newVel);
                    newVel = newVel * updateData.angularVel * Const::DEG_TO_RAD;
                    cell->vel = newVel;
                }
            }
        }
    }

    {
        auto const partition = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = data.objects.particlePointers.at(index);
            if (particle->selected != 0) {
                auto relPos = particle->pos - center;
                data.cellMap.correctDirection(relPos);
                particle->pos = Math::applyMatrix(relPos, rotationMatrix) + center;
                data.cellMap.correctPosition(particle->pos);
            }
        }
    }
}

__global__ void cudaCalcAccumulatedCenterAndVel(SimulationData data, int refCellIndex, float2* center, float2* velocity, int* numEntities, bool includeClusters)
{
    {
        float2 refPos = refCellIndex != -1 ? data.objects.cellPointers.at(refCellIndex)->pos : float2{0, 0};

        auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.objects.cellPointers.at(index);
            if (isSelected(cell, includeClusters)) {
                if (center) {
                    auto pos = cell->pos + data.cellMap.getCorrectionIncrement(refPos, cell->pos);
                    atomicAdd(&center->x, pos.x);
                    atomicAdd(&center->y, pos.y);
                }
                if (velocity) {
                    atomicAdd(&velocity->x, cell->vel.x);
                    atomicAdd(&velocity->y, cell->vel.y);
                }
                atomicAdd(numEntities, 1);
            }
        }
    }
    {
        auto const partition = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = data.objects.particlePointers.at(index);
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
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (isSelected(cell, updateData.considerClusters)) {
            cell->pos = cell->pos + float2{updateData.posDeltaX, updateData.posDeltaY};
            data.cellMap.correctPosition(cell->pos);
            cell->vel = float2{updateData.velX, updateData.velY};
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->pos = particle->pos + float2{updateData.posDeltaX, updateData.posDeltaY};
            data.particleMap.correctPosition(particle->pos);
            particle->vel = float2{updateData.velX, updateData.velY};
        }
    }
}

__global__ void cudaSetVelocityForSelection(SimulationData data, float2 velocity, bool includeClusters)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            cell->vel = velocity;
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->vel = velocity;
        }
    }
}

__global__ void cudaMakeSticky(SimulationData data, bool includeClusters)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            //#TODO introduce sticky flag
        }
    }
}

__global__ void cudaRemoveStickiness(SimulationData data, bool includeClusters)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            //#TODO introduce sticky flag
        }
    }
}

__global__ void cudaSetBarrier(SimulationData data, bool value, bool includeClusters)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            cell->barrier = value;
        }
    }
}

__global__ void cudaScheduleDisconnectSelectionFromRemainings(SimulationData data, int* result)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (1 == cell->selected) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;

                if (1 != connectedCell->selected
                    && data.cellMap.getDistance(cell->pos, connectedCell->pos) > cudaSimulationParameters.maxBindingDistance.value[cell->color]) {
                    CellConnectionProcessor::scheduleDeleteConnectionPair(data, cell, connectedCell);
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
    CellConnectionProcessor::processDeleteConnectionOperations(data);
}

__global__ void cudaProcessAddConnectionChanges(SimulationData data)
{
    CellConnectionProcessor::processAddOperations(data);
}

__global__ void cudaExistsSelection(PointSelectionData pointData, SimulationData data, int* result)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (1 == cell->selected && data.cellMap.getDistance(pointData.pos, cell->pos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (1 == particle->selected && data.cellMap.getDistance(pointData.pos, particle->pos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }
}

__global__ void cudaSetSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (data.cellMap.getDistance(pos, cell->pos) < radius) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (data.particleMap.getDistance(pos, particle->pos) < radius) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSetSelection(AreaSelectionData selectionData, SimulationData data)
{
    auto const cellPartition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);

        if (Math::isInBetweenModulo(toFloat(selectionData.startPos.x), toFloat(selectionData.endPos.x), cell->pos.x, toFloat(data.worldSize.x))
            && Math::isInBetweenModulo(toFloat(selectionData.startPos.y), toFloat(selectionData.endPos.y), cell->pos.y, toFloat(data.worldSize.y))) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particlePartition = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (Math::isInBetweenModulo(toFloat(selectionData.startPos.x), toFloat(selectionData.endPos.x), particle->pos.x, toFloat(data.worldSize.x))
            && Math::isInBetweenModulo(toFloat(selectionData.startPos.y), toFloat(selectionData.endPos.y), particle->pos.y, toFloat(data.worldSize.y))) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaRemoveSelection(SimulationData data, bool onlyClusterSelection)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (!onlyClusterSelection || cell->selected == 2) {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (!onlyClusterSelection || particle->selected == 2) {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSwapSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (data.cellMap.getDistance(pos, cell->pos) < radius) {
            if (cell->selected == 0) {
                cell->selected = 1;
            } else if (cell->selected == 1) {
                cell->selected = 0;
            }
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (data.particleMap.getDistance(pos, particle->pos) < radius) {
            particle->selected = 1 - particle->selected;
        }
    }
}

__global__ void cudaRolloutSelectionStep(SimulationData data, int* result)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);

        if (0 != cell->selected) {
            auto currentCell = cell;

            //heuristics to cover connected cells
            for (int i = 0; i < 30; ++i) {
                bool found = false;
                for (int j = 0; j < currentCell->numConnections; ++j) {
                    auto candidateCell = currentCell->connections[j].cell;
                    if (0 == candidateCell->selected) {
                        currentCell = candidateCell;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    break;
                }

                currentCell->selected = 2;
                atomicExch(result, 1);
            }
        }
    }
}

__global__ void cudaApplyForce(SimulationData data, ApplyForceData applyData)
{
    {
        auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.objects.cellPointers.at(index);
            auto pos = cell->pos;
            pos += data.cellMap.getCorrectionIncrement(applyData.startPos, pos);
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius && !cell->barrier) {
                auto weightedForce = applyData.force;
                //*(actionRadius - distanceToSegment) / actionRadius;
                cell->vel = cell->vel + weightedForce;
            }
        }
    }
    {
        auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

        for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
            auto const& particle = data.objects.particlePointers.at(index);
            auto const& pos = particle->pos;
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius) {
                auto weightedForce = applyData.force;  //*(actionRadius - distanceToSegment) / actionRadius;
                particle->vel = particle->vel + weightedForce;
            }
        }
    }
}

__global__ void cudaResetSelectionResult(SelectionResult result)
{
    result.reset();
}

__global__ void cudaCalcCellWithMinimalPosY(SimulationData data, unsigned long long int* minCellPosYAndIndex)
{
    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (0 != cell->selected) {
            atomicMin(minCellPosYAndIndex, (static_cast<unsigned long long int>(abs(cell->pos.y)) << 32) | static_cast<unsigned long long int>(index));
        }
    }
}

__global__ void cudaGetSelectionShallowData(SimulationData data, int refCellIndex, SelectionResult result)
{
    float2 refPos = refCellIndex != 0xffffffff ? data.objects.cellPointers.at(refCellIndex)->pos : float2{0, 0};

    auto const cellBlock = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (0 != cell->selected) {
            result.collectCell(cell, refPos, data.cellMap);
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.objects.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.objects.particlePointers.at(index);
        if (0 != particle->selected) {
            result.collectParticle(particle, refPos, data.cellMap);
        }
    }
}

__global__ void cudaFinalizeSelectionResult(SelectionResult result, BaseMap map)
{
    result.finalize(map, !cudaSimulationParameters.borderlessRendering.value);
}

__global__ void cudaSetDetached(SimulationData data, bool value)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = data.objects.cellPointers.at(index);
        if (0 != cell->selected) {
            cell->detached = value ? 1 : 0;
        }
    }
}

__global__ void cudaApplyCataclysm(SimulationData data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->cellType == CellType_Constructor) {
            if (data.numberGen1.random() < 0.3f) {
                for (int j = 0; j < 100; ++j) {
                    MutationProcessor::neuronDataMutation(data, cell);
                }
                for (int j = 0; j < 50; ++j) {
                    MutationProcessor::propertiesMutation(data, cell);
                }
                MutationProcessor::geometryMutation(data, cell);
                MutationProcessor::customGeometryMutation(data, cell);
                MutationProcessor::cellTypeMutation(data, cell);
                int num = data.numberGen1.random(5);
                for (int i = 0; i < num; ++i) {
                    MutationProcessor::insertMutation(data, cell);
                }
                //                MutationProcessor::translateMutation(data, cell);
                for (int i = 0; i < 2; ++i) {
                    MutationProcessor::duplicateMutation(data, cell);
                }
                //                MutationProcessor::deleteMutation(data, cell);
            }
        }
    }
}
