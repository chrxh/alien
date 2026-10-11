#pragma once

#include "SimulationData.cuh"

__global__ void cudaApplyForceFields(SimulationData data);