#include "EditKernels.cuh"

__global__ void cudaColorSelectedCells(SimulationData data, unsigned char color, bool includeClusters)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if ((0 != cell->selected && includeClusters) || (1 == cell->selected && !includeClusters)) {
            cell->metadata.color = color;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->metadata.color = color;
        }
    }
}

__global__ void cudaPrepareForUpdate(SimulationData data)
{
    data.prepareForNextTimestep();
}

//assumes that *changeDataTO.numCells == 1
__global__ void cudaChangeCell(SimulationData data, DataAccessTO changeDataTO)
{
    //delete tokens on cell to be changed
    {
        auto const partition = calcAllThreadsPartition(data.entities.tokenPointers.getNumOrigEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& token = data.entities.tokenPointers.at(index);
            if (token->cell->id == changeDataTO.cells[0].id) {
                token = nullptr;
            }
        }
    }
    {
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.entities.cellPointers.at(index);
            auto const& cellTO = changeDataTO.cells[0];
            if (cell->id == cellTO.id) {
                EntityFactory entityFactory;
                entityFactory.init(&data);
                entityFactory.changeCellFromTO(cellTO, changeDataTO, cell);

                for (int i = 0; i < *changeDataTO.numTokens; ++i) {
                    entityFactory.createTokenFromTO(changeDataTO.tokens[i], cell);
                }
            }
        }
    }
}

//assumes that *changeDataTO.numParticles == 1
__global__ void cudaChangeParticle(SimulationData data, DataAccessTO changeDataTO)
{
    auto const partition = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        auto const& particleTO = changeDataTO.particles[0];
        if (particle->id == particleTO.id) {
            EntityFactory entityFactory;
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
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = data.entities.cellPointers.at(index);
            if (isSelected(cell, includeClusters)) {
                cell = nullptr;
            }
        }
    }
    {
        auto const partition = calcAllThreadsPartition(data.entities.tokenPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& token = data.entities.tokenPointers.at(index);
            auto const& cell = token->cell; 
            if (isSelected(cell, includeClusters)) {
                token = nullptr;
            }
        }
    }
    {
        auto const partition = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = data.entities.particlePointers.at(index);
            if (particle->selected == 1) {
                particle = nullptr;
            }
        }
    }
}

__global__ void cudaRemoveSelectedCellConnections(SimulationData data, bool includeClusters, int* retry)
{
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.entities.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                if ((includeClusters && connectedCell->selected == 0) || (!includeClusters && connectedCell->selected != 1)) {

                    if (connectedCell->tryLock()) {
                        CellConnectionProcessor::delConnections(cell, connectedCell);
                        --i;
                        connectedCell->releaseLock();
                    } else {
                        atomicExch(retry, 1);
                    }
                }
            }
        }
    }
}

__global__ void cudaRelaxSelectedEntities(SimulationData data, bool includeClusters)
{
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.entities.cellPointers.at(index);
        if (isSelected(cell, includeClusters)) {
            auto const numConnections = cell->numConnections;
            for (int i = 0; i < numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                if (isSelected(connectedCell, includeClusters)) {
                    auto delta = connectedCell->absPos - cell->absPos;
                    data.cellMap.mapDisplacementCorrection(delta);
                    cell->connections[i].distance = Math::length(delta);
                }
            }

            if (numConnections > 1) {
                for (int i = 0; i < numConnections; ++i) {
                    auto prevConnectedCell = cell->connections[(i + numConnections - 1) % numConnections].cell;
                    auto connectedCell = cell->connections[i].cell;
                    if (isSelected(connectedCell, includeClusters) && isSelected(prevConnectedCell, includeClusters)) {
                        auto prevDisplacement = prevConnectedCell->absPos - cell->absPos;
                        data.cellMap.mapDisplacementCorrection(prevDisplacement);
                        auto prevAngle = Math::angleOfVector(prevDisplacement);

                        auto displacement = connectedCell->absPos - cell->absPos;
                        data.cellMap.mapDisplacementCorrection(displacement);
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
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    Cell* otherCells[18];
    int numOtherCells;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.entities.cellPointers.at(index);
        if (1 != cell->selected) {
            continue;
        }
        data.cellMap.get(otherCells, 18, numOtherCells, cell->absPos, cudaSimulationParameters.cellMaxCollisionDistance);
        for (int i = 0; i < numOtherCells; ++i) {
            Cell* otherCell = otherCells[i];

            if (!otherCell || otherCell == cell) {
                continue;
            }

            if (1 == otherCell->selected && !considerWithinSelection) {
                continue;
            }

            auto posDelta = cell->absPos - otherCell->absPos;
            data.cellMap.mapDisplacementCorrection(posDelta);

            bool alreadyConnected = false;
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (connectedCell == otherCell) {
                    alreadyConnected = true;
                    break;
                }
            }
            if (alreadyConnected) {
                continue;
            }

            if (cell->numConnections < cell->maxConnections && otherCell->numConnections < otherCell->maxConnections) {
                CellConnectionProcessor::scheduleAddConnections(data, cell, otherCell, false);
                atomicExch(result, 1);
            }
        }
    }
}

__global__ void cudaUpdateMapForConnection(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.updateMap(data);
}

__global__ void cudaUpdateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center)
{
    __shared__ Math::Matrix rotationMatrix;
    if (0 == threadIdx.x) {
        Math::rotationMatrix(updateData.angleDelta, rotationMatrix);
    }
    __syncthreads();

    {
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.entities.cellPointers.at(index);
            if ((updateData.considerClusters && cell->selected != 0) || (!updateData.considerClusters && cell->selected == 1)) {
                auto relPos = cell->absPos - center;
                data.cellMap.mapDisplacementCorrection(relPos);

                if (updateData.angleDelta != 0) {
                    cell->absPos = Math::applyMatrix(relPos, rotationMatrix) + center;
                    data.cellMap.mapPosCorrection(cell->absPos);
                }

                if (updateData.angularVelDelta != 0) {
                    auto velDelta = relPos;
                    Math::rotateQuarterClockwise(velDelta);
                    velDelta = velDelta * updateData.angularVelDelta * DEG_TO_RAD;
                    cell->vel = cell->vel + velDelta;
                }
            }
        }
    }

    {
        auto const partition = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = data.entities.particlePointers.at(index);
            if (particle->selected != 0) {
                auto relPos = particle->absPos - center;
                data.cellMap.mapDisplacementCorrection(relPos);
                particle->absPos = Math::applyMatrix(relPos, rotationMatrix) + center;
                data.cellMap.mapPosCorrection(particle->absPos);
            }
        }
    }
}

__global__ void cudaCalcAccumulatedCenter(ShallowUpdateSelectionData updateData, SimulationData data, float2* center, int* numEntities)
{
    {
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.entities.cellPointers.at(index);
            if ((updateData.considerClusters && cell->selected != 0) || (!updateData.considerClusters && cell->selected == 1)) {
                atomicAdd(&center->x, cell->absPos.x);
                atomicAdd(&center->y, cell->absPos.y);
                atomicAdd(numEntities, 1);
            }
        }
    }
    {
        auto const partition = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& particle = data.entities.particlePointers.at(index);
            if (particle->selected != 0) {
                atomicAdd(&center->x, particle->absPos.x);
                atomicAdd(&center->y, particle->absPos.y);
                atomicAdd(numEntities, 1);
            }
        }
    }
}

__global__ void cudaUpdatePosAndVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if ((0 != cell->selected && updateData.considerClusters) || (1 == cell->selected && !updateData.considerClusters)) {
            cell->absPos = cell->absPos + float2{updateData.posDeltaX, updateData.posDeltaY};
            cell->vel = cell->vel + float2{updateData.velDeltaX, updateData.velDeltaY};
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->absPos = particle->absPos + float2{updateData.posDeltaX, updateData.posDeltaY};
            particle->vel = particle->vel + float2{updateData.velDeltaX, updateData.velDeltaY};
        }
    }
}

__global__ void cudaScheduleDisconnectSelectionFromRemainings(SimulationData data, int* result)
{
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (1 == cell->selected) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;

                if (1 != connectedCell->selected
                    && data.cellMap.mapDistance(cell->absPos, connectedCell->absPos) > cudaSimulationParameters.cellMaxBindingDistance) {
                    CellConnectionProcessor::scheduleDelConnection(data, cell, connectedCell);
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

__global__ void cudaProcessConnectionChanges(SimulationData data)
{
    CellConnectionProcessor::processConnectionsOperations(data);
}

__global__ void cudaExistsSelection(PointSelectionData pointData, SimulationData data, int* result)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (1 == cell->selected && data.cellMap.mapDistance(pointData.pos, cell->absPos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (1 == particle->selected && data.cellMap.mapDistance(pointData.pos, particle->absPos) < pointData.radius) {
            atomicExch(result, 1);
        }
    }
}

__global__ void cudaSetSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (data.cellMap.mapDistance(pos, cell->absPos) < radius) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (data.particleMap.mapDistance(pos, particle->absPos) < radius) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSetSelection(AreaSelectionData selectionData, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (isContainedInRect(selectionData.startPos, selectionData.endPos, cell->absPos)) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (isContainedInRect(selectionData.startPos, selectionData.endPos, particle->absPos)) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void cudaRemoveSelection(SimulationData data, bool onlyClusterSelection)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (!onlyClusterSelection || cell->selected == 2) {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (!onlyClusterSelection || particle->selected == 2) {
            particle->selected = 0;
        }
    }
}

__global__ void cudaSwapSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (data.cellMap.mapDistance(pos, cell->absPos) < radius) {
            if (cell->selected == 0) {
                cell->selected = 1;
            } else if (cell->selected == 1) {
                cell->selected = 0;
            }
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (data.particleMap.mapDistance(pos, particle->absPos) < radius) {
            particle->selected = 1 - particle->selected;
        }
    }
}

__global__ void cudaRolloutSelectionStep(SimulationData data, int* result)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);

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
        auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

        for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
            auto const& cell = data.entities.cellPointers.at(index);
            auto const& pos = cell->absPos;
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius) {
                auto weightedForce = applyData.force;
                //*(actionRadius - distanceToSegment) / actionRadius;
                cell->vel = cell->vel + weightedForce;
            }
        }
    }
    {
        auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

        for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
            auto const& particle = data.entities.particlePointers.at(index);
            auto const& pos = particle->absPos;
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

__global__ void cudaGetSelectionShallowData(SimulationData data, SelectionResult result)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (0 != cell->selected) {
            result.collectCell(cell);
        }
    }

    auto const particleBlock = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (0 != particle->selected) {
            result.collectParticle(particle);
        }
    }
}

__global__ void cudaFinalizeSelectionResult(SelectionResult result)
{
    result.finalize();
}
