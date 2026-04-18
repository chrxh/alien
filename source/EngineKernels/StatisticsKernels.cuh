#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

__global__ void cudaUpdateTimestepStatistics_substep1(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateTimestepStatistics_substep2(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateTimestepStatistics_substep3(SimulationData data, SimulationStatistics statistics);

__global__ void cudaUpdateHistogramData_substep1(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateHistogramData_substep2(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateHistogramData_substep3(SimulationData data, SimulationStatistics statistics);
