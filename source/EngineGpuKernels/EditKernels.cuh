#pragma once

#include <EngineInterface/Colors.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>
#include <EngineInterface/SimulationParameters.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "CellProcessor.cuh"
#include "GarbageCollectorKernels.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "SelectionResult.cuh"
#include "SimulationData.cuh"
#include "TO.cuh"

__global__ void cudaColorSelectedCells(SimulationData data, unsigned char color, bool includeClusters);
__global__ void cudaChangeCell(SimulationData data, TO changeTO);      // changeTO contains only 1 cell
__global__ void cudaChangeParticle(SimulationData data, TO changeTO);  // changeTO contains only 1 particle

__global__ void cudaAddGenomeAndCreature(SimulationData data, TO to, Genome** newGenome, Creature** newCreature);
__global__ void cudaChangeCellToCreature(SimulationData data, Creature** newCreature, bool* result);

__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters);
__global__ void cudaRemoveSelectedCellConnections(SimulationData data, bool includeClusters);
__global__ void cudaRelaxSelectedEntities(SimulationData data, bool includeClusters);
__global__ void cudaScheduleConnectSelection(SimulationData data, bool considerWithinSelection, int* result);
__global__ void cudaPrepareMapForReconnection(SimulationData data);
__global__ void cudaUpdateMapForReconnection(SimulationData data);
__global__ void cudaUpdateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center);
__global__ void
cudaCalcAccumulatedCenterAndVel(SimulationData data, int refCellIndex, float2* center, float2* velocity, int* numEntities, bool includeClusters);
__global__ void cudaIncrementPosAndVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data);
__global__ void cudaSetVelocityForSelection(SimulationData data, float2 velocity, bool includeClusters);
__global__ void cudaMakeSticky(SimulationData data, bool includeClusters);
__global__ void cudaRemoveStickiness(SimulationData data, bool includeClusters);
__global__ void cudaSetBarrier(SimulationData data, bool value, bool includeClusters);
__global__ void cudaScheduleDisconnectSelectionFromRemainings(SimulationData data, int* result);
__global__ void cudaPrepareConnectionChanges(SimulationData data);
__global__ void cudaProcessDeleteConnectionChanges(SimulationData data);
__global__ void cudaProcessAddConnectionChanges(SimulationData data);
__global__ void cudaApplyForce(SimulationData data, ApplyForceData applyData);
__global__ void cudaSetDetached(SimulationData data, bool value);
__global__ void cudaApplyCataclysm(SimulationData data);
