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

__global__ void cudaDrawBackground(uint64_t* imageData, int2 imageSize, int2 worldSize, float zoom, float2 rectUpperLeft, float2 rectLowerRight);
__global__ void
cudaDrawCells(int2 universeSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Cell*> cells, uint64_t* imageData, int2 imageSize, float zoom);
__global__ void
cudaDrawTokens(int2 universeSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Token*> tokens, uint64_t* imageData, int2 imageSize, float zoom);
__global__ void
cudaDrawParticles(int2 universeSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Particle*> particles, uint64_t* imageData, int2 imageSize, float zoom);
__global__ void cudaDrawFlowCenters(uint64_t* targetImage, float2 rectUpperLeft, int2 imageSize, float zoom);
