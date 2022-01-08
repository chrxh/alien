#pragma once

#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

__device__ extern float getHeight(float2 const& pos, MapInfo const& mapInfo);

__device__ extern float2 calcVelocity(float2 const& pos, MapInfo const& mapInfo);

__global__ extern void applyFlowFieldSettings(SimulationData data);

__global__ extern void cudaApplyFlowFieldSettings(SimulationData data);