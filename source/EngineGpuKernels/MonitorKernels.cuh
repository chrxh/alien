#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CudaMonitorData.cuh"

__global__ void cudaGetCudaMonitorData_substep1(SimulationData data, CudaMonitorData monitorData);
__global__ void cudaGetCudaMonitorData_substep2(SimulationData data, CudaMonitorData monitorData);
