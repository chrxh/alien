#pragma once

#include <EngineInterface/MutationType.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t objectId, MutationType mutationType);
__global__ void cudaTestMutationCheck(SimulationData data, uint64_t objectId);
__global__ void cudaTestMutate(SimulationData data, uint64_t objectId, MutationType mutationType);
__global__ void cudaTestCreateConnection(SimulationData data, uint64_t objectId1, uint64_t objectId2);
__global__ void cudaTestArePointersValid(SimulationData data, bool* result);
