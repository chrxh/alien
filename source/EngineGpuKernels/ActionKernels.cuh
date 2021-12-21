#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "CleanupKernels.cuh"
#include "SelectionResult.cuh"
#include "CellConnectionProcessor.cuh"
#include "CellProcessor.cuh"

#include "SimulationData.cuh"

__global__ void applyForceToCells(ApplyForceData applyData, int2 universeSize, Array<Cell*> cells)
{
    auto const cellBlock = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = cells.at(index);
        auto const& pos = cell->absPos;
        auto distanceToSegment =
            Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
        if (distanceToSegment < applyData.radius) {
            auto weightedForce = applyData.force;
            //*(actionRadius - distanceToSegment) / actionRadius;
            cell->vel = cell->vel + weightedForce;
        }
    }
}

__global__ void applyForceToParticles(ApplyForceData applyData, int2 universeSize, Array<Particle*> particles)
{
    auto const particleBlock = calcAllThreadsPartition(particles.getNumEntries());

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);
        auto const& pos = particle->absPos;
        auto distanceToSegment =
            Math::calcDistanceToLineSegment(applyData.startPos, applyData.endPos, pos, applyData.radius);
        if (distanceToSegment < applyData.radius) {
            auto weightedForce = applyData.force;//*(actionRadius - distanceToSegment) / actionRadius;
            particle->vel = particle->vel + weightedForce;
        }
    }
}

__global__ void existSelection(PointSelectionData pointData, SimulationData data, int* result)
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

__global__ void setSelection(float2 pos, float radius, SimulationData data)
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

__global__ void setSelection(AreaSelectionData selectionData, SimulationData data)
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

__global__ void swapSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (data.cellMap.mapDistance(pos, cell->absPos) < radius) {
            if (cell->selected == 0) {
                cell->selected = 1;
            }
            else if (cell->selected == 1) {
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

__global__ void rolloutSelectionStep(SimulationData data, int* result)
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

__global__ void rolloutSelection(SimulationData data)
{
    int* result = new int;
    do {
        *result = 0;
        KERNEL_CALL(rolloutSelectionStep, data, result);
    } while (1 == *result);
    delete result;
}

__global__ void updatePosAndVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if ((0 != cell->selected && updateData.considerClusters)
            || (1 == cell->selected && !updateData.considerClusters)) {
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

__global__ void removeSelection(SimulationData data, bool onlyClusterSelection)
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

__global__ void removeClusterSelection(SimulationData data)
{
    auto const cellBlock = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (2 == cell->selected) {
            cell->selected = 0;
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

__global__ void disconnectSelection(SimulationData data, int* result)
{
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (1 == cell->selected) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                
                if (1 != connectedCell->selected
                    && data.cellMap.mapDistance(cell->absPos, connectedCell->absPos)
                        > cudaSimulationParameters.cellMaxBindingDistance) {
                    CellConnectionProcessor::scheduleDelConnection(data, cell, connectedCell);
                    atomicExch(result, 1);
                }
            }
        }
    }
}

__global__ void updateMapForConnection(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.updateMap(data);
}

__global__ void connectSelection(SimulationData data, int* result)
{
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    Cell* otherCells[18];
    int numOtherCells;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.entities.cellPointers.at(index);
        if (1 != cell->selected) {
            continue;
        }
        data.cellMap.get(
            otherCells, 18, numOtherCells, cell->absPos, cudaSimulationParameters.cellMaxCollisionDistance);
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

__global__ void processConnectionChanges(SimulationData data)
{
    CellConnectionProcessor::processConnectionsOperations(data);
}

__global__ void calcAccumulatedCenter(ShallowUpdateSelectionData updateData, SimulationData data, float2* center, int* numEntities)
{
    {
        auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto const& cell = data.entities.cellPointers.at(index);
            if ((updateData.considerClusters && cell->selected != 0)
                || (!updateData.considerClusters && cell->selected == 1)) {
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

__global__ void
updateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center)
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
            if ((updateData.considerClusters && cell->selected != 0)
                || (!updateData.considerClusters && cell->selected == 1)) {
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

__global__ void removeSelectedCellConnections(SimulationData data, bool includeClusters, int* retry)
{
    auto const partition = calcAllThreadsPartition(data.entities.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.entities.cellPointers.at(index);
        if ((includeClusters && cell->selected != 0) || (!includeClusters && cell->selected == 1)) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                if ((includeClusters && connectedCell->selected == 0)
                    || (!includeClusters && connectedCell->selected != 1)) {

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

/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ void cudaApplyForce(ApplyForceData applyData, SimulationData data)
{
    KERNEL_CALL(applyForceToCells, applyData, data.size, data.entities.cellPointers);
    KERNEL_CALL(applyForceToParticles, applyData, data.size, data.entities.particlePointers);
}

__global__ void
cudaSwitchSelection(PointSelectionData switchData, SimulationData data)
{
    int* result = new int;
    *result = 0; 

    KERNEL_CALL(existSelection, switchData, data, result);
    if (0 == *result) {
        KERNEL_CALL(setSelection, switchData.pos, switchData.radius, data);
        KERNEL_CALL_1_1(rolloutSelection, data);
    }

    delete result;
}

__global__ void cudaSwapSelection(PointSelectionData switchData, SimulationData data)
{
    int* result = new int;
    *result = 0;

    KERNEL_CALL(removeSelection, data, true);

    KERNEL_CALL(swapSelection, switchData.pos, switchData.radius, data);
    KERNEL_CALL_1_1(rolloutSelection, data);

    delete result;
}

__global__ void cudaSetSelection(AreaSelectionData setData, SimulationData data)
{
    int* result = new int;
    *result = 0;

    KERNEL_CALL(setSelection, setData, data);
    KERNEL_CALL_1_1(rolloutSelection, data);

    delete result;
}

__global__ void cudaGetSelectionShallowData(SimulationData data, SelectionResult selectionResult)
{
    selectionResult.reset();
    KERNEL_CALL(getSelectionShallowData, data, selectionResult);
    selectionResult.finalize();
}

__global__ void cudaShallowUpdateSelection(ShallowUpdateSelectionData updateData, SimulationData data)
{
    int* result = new int;

    bool reconnectionRequired =
        !updateData.considerClusters && (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.angleDelta != 0);

    //disconnect selection in case of reconnection
    if (reconnectionRequired) {
        int counter = 10;
        do {
            *result = 0;
            data.prepareForSimulation();
            KERNEL_CALL(disconnectSelection, data, result);
            KERNEL_CALL(processConnectionChanges, data);
        } while (1 == *result && --counter > 0);    //due to locking not all affecting connections may be removed at first => repeat
    }

    if (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.velDeltaX != 0
        || updateData.velDeltaY != 0) {
        KERNEL_CALL(updatePosAndVelForSelection, updateData, data);
    }
    if (updateData.angleDelta != 0 || updateData.angularVelDelta != 0) {
        float2* center = new float2;
        int* numEntities = new int;
        *center = {0, 0};
        *numEntities = 0;
        KERNEL_CALL(calcAccumulatedCenter, updateData, data, center, numEntities);
        if (*numEntities != 0) {
            *center = *center / *numEntities;
        }
        KERNEL_CALL(updateAngleAndAngularVelForSelection, updateData, data, *center);

        delete center;
        delete numEntities;
    }

    //connect selection in case of reconnection
    if (reconnectionRequired) {

        int counter = 10;
        do {
            *result = 0;
            data.prepareForSimulation();

            KERNEL_CALL(updateMapForConnection, data);
            KERNEL_CALL(connectSelection, data, result);
            KERNEL_CALL(processConnectionChanges, data);

            KERNEL_CALL(cleanupCellMap, data);
        } while (1 == *result && --counter > 0);    //due to locking not all necessary connections may be established at first => repeat

        //update selection
        KERNEL_CALL(removeClusterSelection, data);
        KERNEL_CALL_1_1(rolloutSelection, data);
    }

    delete result;
}

__global__ void cudaRemoveSelection(SimulationData data)
{
    KERNEL_CALL(removeSelection, data, false);
}

__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters)
{
    int* result = new int;

    do {
        *result = 0;
        KERNEL_CALL(removeSelectedCellConnections, data, includeClusters, result);
    } while (1 == *result);
    KERNEL_CALL(removeSelectedCells, data, includeClusters);
    KERNEL_CALL(removeSelectedParticles, data);
    KERNEL_CALL_1_1(cleanupAfterDataManipulationKernel, data);

    delete result;
}
