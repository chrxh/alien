#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "SelectionResult.cuh"
#include "CellConnectionProcessor.cuh"
#include "CellProcessor.cuh"

#include "SimulationData.cuh"

__global__ void cudaColorSelectedCells(SimulationData data, unsigned char color, bool includeClusters);
__global__ void cudaPrepareForUpdate(SimulationData data);
__global__ void cudaChangeCell(SimulationData data, DataAccessTO changeDataTO);  //assumes that *changeDataTO.numCells == 1
__global__ void cudaChangeParticle(SimulationData data, DataAccessTO changeDataTO); //assumes that *changeDataTO.numParticles == 1
__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters);
__global__ void cudaRemoveSelectedCellConnections(SimulationData data, bool includeClusters, int* retry);
__global__ void cudaRelaxSelectedEntities(SimulationData data, bool includeClusters);
__global__ void cudaScheduleConnectSelection(SimulationData data, bool considerWithinSelection, int* result);
__global__ void cudaUpdateMapForConnection(SimulationData data);
__global__ void cudaUpdateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center);
__global__ void cudaCalcAccumulatedCenterAndVel(SimulationData data, float2* center, float2* velocity, int* numEntities, bool includeClusters);
__global__ void cudaIncrementPosAndVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data);
__global__ void cudaSetVelocityForSelection(SimulationData data, float2 velocity, bool includeClusters);
__global__ void cudaRemoveStickiness(SimulationData data, bool includeClusters);
__global__ void cudaScheduleDisconnectSelectionFromRemainings(SimulationData data, int* result);
__global__ void cudaPrepareConnectionChanges(SimulationData data);
__global__ void cudaProcessConnectionChanges(SimulationData data);
__global__ void cudaExistsSelection(PointSelectionData pointData, SimulationData data, int* result);
__global__ void cudaSetSelection(float2 pos, float radius, SimulationData data);
__global__ void cudaSetSelection(AreaSelectionData selectionData, SimulationData data);
__global__ void cudaRemoveSelection(SimulationData data, bool onlyClusterSelection);
__global__ void cudaSwapSelection(float2 pos, float radius, SimulationData data);
__global__ void cudaRolloutSelectionStep(SimulationData data, int* result);
__global__ void cudaApplyForce(SimulationData data, ApplyForceData applyData);
__global__ void cudaResetSelectionResult(SelectionResult result);
__global__ void cudaGetSelectionShallowData(SimulationData data, SelectionResult result);
__global__ void cudaFinalizeSelectionResult(SelectionResult result);
