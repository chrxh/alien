#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/ZoomLevels.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"
#include "RenderingData.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ extern void
cudaDrawImageKernel(float2 rectUpperLeft, float2 rectLowerRight, int2 imageSize, float zoom, SimulationData data, RenderingData renderingData);
