#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"

__global__ void cudaMutateNeuronData(SimulationData data, uint64_t cellId);
__global__ void cudaMutateData(SimulationData data, uint64_t cellId);
