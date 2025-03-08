#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/ZoomLevels.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"
#include "ObjectFactory.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"
#include "RenderingData.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void cudaDrawSpotsAndGridlines(uint64_t* imageData, int2 imageSize, int2 worldSize, float zoom, float2 rectUpperLeft, float2 rectLowerRight);
__global__ void cudaPrepareFilteringForRendering(Array<Cell*> filteredCells, Array<Particle*> filteredParticles);
__global__ void cudaFilterCellsForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Cell*> cells,
    Array<Cell*> filteredCells,
    int2 imageSize,
    float zoom);
__global__ void cudaFilterParticlesForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Particle*> particles,
    Array<Particle*> filteredParticles,
    int2 imageSize,
    float zoom);
__global__ void cudaDrawCells(
    uint64_t timestep,
    int2 worldSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Cell*> cells,
    uint64_t* imageData,
    int2 imageSize,
    float zoom);
__global__ void cudaDrawCellGlow(int2 worldSize, float2 rectUpperLeft, Array<Cell*> cells, uint64_t* imageData, int2 imageSize, float zoom);

__global__ void cudaDrawParticles(int2 worldSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Particle*> particles, uint64_t* imageData, int2 imageSize, float zoom);
__global__ void cudaDrawRadiationSources(uint64_t* targetImage, float2 rectUpperLeft, int2 worldSize, int2 imageSize, float zoom);
__global__ void cudaDrawRepetition(int2 worldSize, int2 imageSize, float2 rectUpperLeft, float2 rectLowerRight, uint64_t* imageData, float zoom);
