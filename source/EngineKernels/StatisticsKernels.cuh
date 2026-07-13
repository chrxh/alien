#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

__global__ void cudaUpdateEvolutionStatistics_substep1(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateEvolutionStatistics_substep2(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateEvolutionStatistics_substep3(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateEvolutionStatistics_substep4(SimulationData data, SimulationStatistics statistics);
__global__ void cudaUpdateEvolutionStatistics_substep5(SimulationData data, SimulationStatistics statistics);

__global__ void cudaPrepareLineageAccumulatorGC(SimulationStatistics statistics);
__global__ void cudaLineageAccumulatorGC(SimulationStatistics statistics);
__global__ void cudaFinishLineageAccumulatorGC(SimulationStatistics statistics);

