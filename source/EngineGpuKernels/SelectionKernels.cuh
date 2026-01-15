#pragma once

#include <EngineInterface/Colors.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>
#include <EngineInterface/SimulationParameters.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "ObjectProcessor.cuh"
#include "GarbageCollectorKernels.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "SelectionResult.cuh"
#include "SimulationData.cuh"
#include "TO.cuh"


__global__ void cudaRemoveSelection(SimulationData data, bool onlyClusterSelection);
__global__ void cudaSwapSelection(float2 pos, float radius, SimulationData data);
__global__ void cudaExistsSelection(PointSelectionData pointData, SimulationData data, int* result);
__global__ void cudaSetSelection(float2 pos, float radius, SimulationData data);
__global__ void cudaSetSelection(AreaSelectionData selectionData, SimulationData data);

__global__ void cudaRolloutSelectionStep(SimulationData data, int* result);