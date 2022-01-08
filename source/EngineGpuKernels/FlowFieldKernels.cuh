#pragma once

#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

__global__ extern void cudaApplyFlowFieldSettings(SimulationData data);