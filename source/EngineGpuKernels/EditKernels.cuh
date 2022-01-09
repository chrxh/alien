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

__global__ void cudaRolloutSelection(SimulationData data);
__global__ void cudaApplyForce(ApplyForceData applyData, SimulationData data);
__global__ void cudaSwitchSelection(PointSelectionData switchData, SimulationData data);
__global__ void cudaSwapSelection(PointSelectionData switchData, SimulationData data);
__global__ void cudaSetSelection(AreaSelectionData setData, SimulationData data);
__global__ void cudaGetSelectionShallowData(SimulationData data, SelectionResult selectionResult);
__global__ void cudaUpdateSelection(SimulationData data);
__global__ void cudaShallowUpdateSelectedEntities(ShallowUpdateSelectionData updateData, SimulationData data);
__global__ void cudaRemoveSelection(SimulationData data);
__global__ void cudaRemoveSelectedEntities(SimulationData data, bool includeClusters);
__global__ void cudaColorSelectedEntities(SimulationData data, unsigned char color, bool includeClusters);
__global__ void cudaChangeSimulationData(SimulationData data, DataAccessTO changeDataTO);
