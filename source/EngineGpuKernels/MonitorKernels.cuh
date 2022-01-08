#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CudaMonitorData.cuh"

__global__ extern void cudaGetCudaMonitorData(SimulationData data, CudaMonitorData monitorData);

__global__ extern void getEnergyForMonitorData(SimulationData data, CudaMonitorData monitorData);

