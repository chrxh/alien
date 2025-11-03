#pragma once

#include "Map.cuh"
#include "Math.cuh"
#include "SimulationData.cuh"

__global__ void cudaApplyForceFieldSettings(SimulationData data);