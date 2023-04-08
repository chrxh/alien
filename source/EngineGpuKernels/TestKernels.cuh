#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/MutationType.h"
#include "SimulationData.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t cellId, MutationType mutationType);
