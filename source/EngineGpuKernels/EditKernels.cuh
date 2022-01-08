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

__global__ extern void applyForceToCells(ApplyForceData applyData, int2 universeSize, Array<Cell*> cells);

__global__ extern void applyForceToParticles(ApplyForceData applyData, int2 universeSize, Array<Particle*> particles);

__global__ extern void existSelection(PointSelectionData pointData, SimulationData data, int* result);

__global__ extern void setSelection(float2 pos, float radius, SimulationData data);

__global__ extern void setSelection(AreaSelectionData selectionData, SimulationData data);

__global__ extern void swapSelection(float2 pos, float radius, SimulationData data);

__global__ extern void rolloutSelectionStep(SimulationData data, int* result);

__global__ extern void rolloutSelection(SimulationData data);

__global__ extern void updatePosAndVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data);

__global__ extern void removeSelection(SimulationData data, bool onlyClusterSelection);

__global__ extern void removeClusterSelection(SimulationData data);

__global__ extern void getSelectionShallowData(SimulationData data, SelectionResult result);

__global__ extern void disconnectSelection(SimulationData data, int* result);

__global__ extern void updateMapForConnection(SimulationData data);

__global__ extern void connectSelection(SimulationData data, int* result);

__global__ extern void processConnectionChanges(SimulationData data);

__global__ extern void calcAccumulatedCenter(ShallowUpdateSelectionData updateData, SimulationData data, float2* center, int* numEntities);

__global__ extern void updateAngleAndAngularVelForSelection(ShallowUpdateSelectionData updateData, SimulationData data, float2 center);

__global__ extern void removeSelectedCellConnections(SimulationData data, bool includeClusters, int* retry);

__global__ extern void removeSelectedCells(SimulationData data, bool includeClusters);

__global__ extern void removeSelectedParticles(SimulationData data);

__global__ extern void colorSelection(SimulationData data, unsigned char color, bool includeClusters);

//assumes that *changeDataTO.numCells == 1
__global__ extern void changeCell(SimulationData data, DataAccessTO changeDataTO, int numTokenPointers);

//assumes that *changeDataTO.numCells == 1
__global__ extern void changeParticle(SimulationData data, DataAccessTO changeDataTO);

/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ extern void cudaApplyForce(ApplyForceData applyData, SimulationData data);

__global__ extern void cudaSwitchSelection(PointSelectionData switchData, SimulationData data);

__global__ extern void cudaSwapSelection(PointSelectionData switchData, SimulationData data);

__global__ extern void cudaSetSelection(AreaSelectionData setData, SimulationData data);

__global__ extern void cudaGetSelectionShallowData(SimulationData data, SelectionResult selectionResult);

__global__ extern void cudaUpdateSelection(SimulationData data);

__global__ extern void cudaShallowUpdateSelectedEntities(ShallowUpdateSelectionData updateData, SimulationData data);

__global__ extern void cudaRemoveSelection(SimulationData data);

__global__ extern void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters);

__global__ extern void cudaColorSelectedEntities(SimulationData data, unsigned char color, bool includeClusters);

__global__ extern void cudaChangeSimulationData(SimulationData data, DataAccessTO changeDataTO);
