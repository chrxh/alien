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
    auto const cellBlock =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

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
    auto const particleBlock =
        calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

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

__global__ void existSelection(SwitchSelectionData switchData, SimulationData data, int* result)
{
    auto const cellBlock =
        calcPartition(data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (1 == cell->selected && data.cellMap.mapDistance(switchData.pos, cell->absPos) < switchData.radius) {
            atomicExch(result, 1);
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (1 == particle->selected && data.cellMap.mapDistance(switchData.pos, particle->absPos) < switchData.radius) {
            atomicExch(result, 1);
        }
    }
}

__global__ void setSelection(float2 pos, float radius, SimulationData data)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (data.cellMap.mapDistance(pos, cell->absPos) < radius) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (data.particleMap.mapDistance(pos, particle->absPos) < radius) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void setSelection(SetSelectionData selectionData, SimulationData data)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (isContainedInRect(selectionData.startPos, selectionData.endPos, cell->absPos)) {
            cell->selected = 1;
        } else {
            cell->selected = 0;
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (isContainedInRect(selectionData.startPos, selectionData.endPos, particle->absPos)) {
            particle->selected = 1;
        } else {
            particle->selected = 0;
        }
    }
}

__global__ void rolloutSelection(SimulationData data, int* result)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

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

__global__ void shallowUpdateSelection(ShallowUpdateSelectionData updateData, SimulationData data)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if ((0 != cell->selected && updateData.considerClusters)
            || (1 == cell->selected && !updateData.considerClusters)) {
            cell->absPos = cell->absPos + float2{updateData.posDeltaX, updateData.posDeltaY};
            cell->vel = cell->vel + float2{updateData.velDeltaX, updateData.velDeltaY};
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->absPos = particle->absPos + float2{updateData.posDeltaX, updateData.posDeltaY};
            particle->vel = particle->vel + float2{updateData.velDeltaX, updateData.velDeltaY};
        }
    }
}

__global__ void removeSelection(SimulationData data)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        cell->selected = 0;
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        particle->selected = 0;
    }
}

__global__ void removeClusterSelection(SimulationData data)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (2 == cell->selected) {
            cell->selected = 0;
        }
    }
}

__global__ void getSelectionShallowData(SimulationData data, SelectionResult result)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (0 != cell->selected) {
            result.collectCell(cell);
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (0 != particle->selected) {
            result.collectParticle(particle);
        }
    }
}

__global__ void disconnectSelection(SimulationData data, int* result)
{
    auto const partition = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (1 == cell->selected) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                
                if (1 != connectedCell->selected && data.cellMap.mapDistance(cell->absPos, connectedCell->absPos) > cudaSimulationParameters.cellMaxBindingDistance) {
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
    auto const partition = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

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


/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ void cudaApplyForce(ApplyForceData applyData, SimulationData data)
{
    KERNEL_CALL(applyForceToCells, applyData, data.size, data.entities.cellPointers);
    KERNEL_CALL(applyForceToParticles, applyData, data.size, data.entities.particlePointers);
}

__global__ void
cudaSwitchSelection(SwitchSelectionData switchData, SimulationData data)
{
    int* result = new int;
    *result = 0; 

    KERNEL_CALL(existSelection, switchData, data, result);
    if (0 == *result) {
        KERNEL_CALL(setSelection, switchData.pos, switchData.radius, data);
        do {
            *result = 0;
            KERNEL_CALL(rolloutSelection, data, result);
        } while(1 == *result);
    }

    delete result;
}

__global__ void cudaSetSelection(SetSelectionData setData, SimulationData data)
{
    int* result = new int;
    *result = 0;

    KERNEL_CALL(setSelection, setData, data);
    do {
        *result = 0;
        KERNEL_CALL(rolloutSelection, data, result);
    } while (1 == *result);

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
        !updateData.considerClusters && (updateData.posDeltaX != 0 || updateData.posDeltaY != 0);

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

    KERNEL_CALL(shallowUpdateSelection, updateData, data);

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
        do {
            *result = 0;
            KERNEL_CALL(rolloutSelection, data, result);
        } while (1 == *result);
    }

    delete result;
}

__global__ void cudaRemoveSelection(SimulationData data)
{
    KERNEL_CALL(removeSelection, data);
}
