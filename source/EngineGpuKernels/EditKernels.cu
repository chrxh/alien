#include "EditKernels.cuh"

namespace
{
    __global__ void applyForceToCells(ApplyForceData applyData, Array<Cell*> cells)
    {
        auto const cellBlock = calcAllThreadsPartition(cells.getNumEntries());

        for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
            auto const& cell = cells.at(index);
            auto const& pos = cell->absPos;
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius) {
                auto weightedForce = applyData.force;
                //*(actionRadius - distanceToSegment) / actionRadius;
                cell->vel = cell->vel + weightedForce;
            }
        }
    }

    __global__ void applyForceToParticles(ApplyForceData applyData, Array<Particle*> particles)
    {
        auto const particleBlock = calcAllThreadsPartition(particles.getNumEntries());

        for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
            auto const& particle = particles.at(index);
            auto const& pos = particle->absPos;
            auto distanceToSegment = Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
            if (distanceToSegment < applyData.radius) {
                auto weightedForce = applyData.force;  //*(actionRadius - distanceToSegment) / actionRadius;
                particle->vel = particle->vel + weightedForce;
            }
        }
    }

    __global__ void getSelectionShallowData(SimulationData data, SelectionResult result)
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

    __global__ void removeSelectedCellConnections(SimulationData data, bool includeClusters, int* retry)
    {
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = data.entities.cellPointers.at(index);
            if ((includeClusters && cell->selected != 0) || (!includeClusters && cell->selected == 1)) {
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

    __global__ void removeSelectedCells(SimulationData data, bool includeClusters)
    {
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = data.entities.cellPointers.at(index);
            if ((includeClusters && cell->selected != 0) || (!includeClusters && cell->selected == 1)) {
                cell = nullptr;
            }
        }
    }

    __global__ void removeSelectedParticles(SimulationData data)
    {
        auto const partition = calcAllThreadsPartition(data.entities.particlePointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = data.entities.particlePointers.at(index);
            if (particle->selected == 1) {
                particle = nullptr;
            }
        }
    }

    __global__ void colorSelection(SimulationData data, unsigned char color, bool includeClusters)
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

    //assumes that *changeDataTO.numCells == 1
    __global__ void changeCell(SimulationData data, DataAccessTO changeDataTO, int numTokenPointers)
    {
        //delete tokens on cell to be changed
        {
            auto const partition = calcAllThreadsPartition(numTokenPointers);
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

    //assumes that *changeDataTO.numCells == 1
    __global__ void changeParticle(SimulationData data, DataAccessTO changeDataTO)
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
}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/
__global__ void cudaPrepareForUpdate(SimulationData data)
{
    data.prepareForNextTimestep();
}

__global__ void cudaConnectSelection(SimulationData data, int* result)
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

            if (1 == otherCell->selected) {
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

__global__ void cudaDisconnectSelection(SimulationData data, int* result)
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
            for (int i = 0; i < 10; ++i) {
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

__global__ void cudaApplyForce(ApplyForceData applyData, SimulationData data)
{
    DEPRECATED_KERNEL_CALL_SYNC(applyForceToCells, applyData, data.entities.cellPointers);
    DEPRECATED_KERNEL_CALL_SYNC(applyForceToParticles, applyData, data.entities.particlePointers);
}

__global__ void cudaGetSelectionShallowData(SimulationData data, SelectionResult selectionResult)
{
    selectionResult.reset();
    DEPRECATED_KERNEL_CALL_SYNC(getSelectionShallowData, data, selectionResult);
    selectionResult.finalize();
}

__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters)
{
    int* result = new int;

    do {
        *result = 0;
        DEPRECATED_KERNEL_CALL_SYNC(removeSelectedCellConnections, data, includeClusters, result);
    } while (1 == *result);
    DEPRECATED_KERNEL_CALL_SYNC(removeSelectedCells, data, includeClusters);
    DEPRECATED_KERNEL_CALL_SYNC(removeSelectedParticles, data);
    DEPRECATED_KERNEL_CALL_SYNC_1_1(cleanupAfterDataManipulationKernel, data);

    delete result;
}

__global__ void cudaColorSelectedEntities(SimulationData data, unsigned char color, bool includeClusters)
{
    DEPRECATED_KERNEL_CALL_SYNC(colorSelection, data, color, includeClusters);
}

__global__ void cudaChangeSimulationData(SimulationData data, DataAccessTO changeDataTO)
{
    if (*changeDataTO.numCells == 1) {
        DEPRECATED_KERNEL_CALL_SYNC(changeCell, data, changeDataTO, data.entities.tokenPointers.getNumEntries());
    }
    if (*changeDataTO.numParticles == 1) {
        DEPRECATED_KERNEL_CALL_SYNC(changeParticle, data, changeDataTO);
    }
    DEPRECATED_KERNEL_CALL_SYNC_1_1(cleanupAfterDataManipulationKernel, data);
}
