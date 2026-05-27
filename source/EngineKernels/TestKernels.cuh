#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"

__global__ void cudaTestMutate(SimulationData data, uint64_t objectId);
__global__ void cudaTestCreateConnection(SimulationData data, uint64_t objectId1, uint64_t objectId2);
__global__ void cudaTestCreateConnectionWithAbsAngle(
    SimulationData data,
    uint64_t objectId1,
    uint64_t objectId2,
    float desiredDistance,
    float desiredAbsAngle1,
    float desiredAbsAngle2);
__global__ void cudaTestIsDataValid(SimulationData data, bool* result);
