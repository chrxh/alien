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

__device__ __inline__ void drawPixel(uint64_t* imageData, unsigned int index, float3 const& color);

__device__ __inline__ void drawAddingPixel(uint64_t* imageData, unsigned int index, float3 const& colorToAdd);

__device__ extern float3 colorToFloat3(unsigned int value);

__device__ extern float3 mix(float3 const& a, float3 const& b, float factor);

__device__ extern float3 mix(float3 const& a, float3 const& b, float3 const& c, float factor1, float factor2);

__global__ extern void drawBackground(uint64_t* imageData, int2 imageSize, int2 worldSize, float zoom, float2 rectUpperLeft, float2 rectLowerRight);

__device__ __inline__ float2 mapUniversePosToVectorImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom);

__device__ __inline__ float2 mapUniversePosToPixelImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom);

__device__ __inline__ float3 calcColor(Cell* cell, int selected);

__device__ __inline__ float3 calcColor(Particle* particle, bool selected);

__device__ __inline__ float3 calcColor(Token* token, bool selected);

__device__ __inline__ void drawDot(uint64_t* imageData, int2 const& imageSize, float2 const& pos, float3 const& colorToAdd);

__device__ __inline__ void drawCircle(uint64_t* imageData, int2 const& imageSize, float2 pos, float3 color, float radius, bool inverted = false);

__global__ extern void
drawCells(int2 universeSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Cell*> cells, uint64_t* imageData, int2 imageSize, float zoom);

__global__ extern void
drawTokens(int2 universeSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Token*> tokens, uint64_t* imageData, int2 imageSize, float zoom);

__global__ extern void
drawParticles(int2 universeSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Particle*> particles, uint64_t* imageData, int2 imageSize, float zoom);

__device__ extern void drawFlowCenters(uint64_t* targetImage, float2 const& rectUpperLeft, int2 imageSize, float zoom);

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ extern void
drawImageKernel(float2 rectUpperLeft, float2 rectLowerRight, int2 imageSize, float zoom, SimulationData data, RenderingData renderingData);
