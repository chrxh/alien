#pragma once

#include "EngineInterface/Colors.h"

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "CleanupKernels.cuh"
#include "SelectionResult.cuh"

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

        //--
/*
        bool neighborSelected = false;
        for(int i = 0; i < cell->numConnections; ++i) {
            auto& otherCell = cell->connections[i].cell;
            if (0 != otherCell->selected) {
                neighborSelected = true;
            }
        }
        if (neighborSelected && 0 == cell->selected) {
            cell->selected = 2;
            atomicExch(result, 1);
        }
*/
    }
}

__global__ void moveSelection(float2 displacement, SimulationData data)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (0 != cell->selected) {
            cell->absPos = cell->absPos + displacement;
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (0 != particle->selected) {
            particle->absPos = particle->absPos + displacement;
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

__global__ void getSelection(SimulationData data, SelectionResult result)
{
    auto const cellBlock = calcPartition(
        data.entities.cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = cellBlock.startIndex; index <= cellBlock.endIndex; ++index) {
        auto const& cell = data.entities.cellPointers.at(index);
        if (1 == cell->selected) {
            result.incSelectedCell();
        }
        if (2 == cell->selected) {
            result.incIndirectSelectedCell();
        }
    }

    auto const particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = data.entities.particlePointers.at(index);
        if (1 == particle->selected) {
            result.incSelectedParticle();
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

__global__ void cudaGetSelection(SimulationData data, SelectionResult selectionResult)
{
    selectionResult.reset();
    KERNEL_CALL(getSelection, data, selectionResult);
}

__global__ void cudaMoveSelection(MoveSelectionData moveData, SimulationData data)
{
    KERNEL_CALL(moveSelection, moveData.displacement, data);
}

__global__ void cudaRemoveSelection(SimulationData data)
{
    KERNEL_CALL(removeSelection, data);
}
