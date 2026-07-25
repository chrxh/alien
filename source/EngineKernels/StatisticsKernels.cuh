#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

// Must be called in this order: the lineage slot of a creature is determined in cudaCollectObjectAndCreatureStatistics
// and reused in cudaCollectGenomeAndEnergyStatistics
__global__ void cudaResetStatistics(SimulationData data, SimulationStatistics statistics);
__global__ void cudaCollectObjectAndCreatureStatistics(SimulationData data, SimulationStatistics statistics);
__global__ void cudaCollectGenomeAndEnergyStatistics(SimulationData data, SimulationStatistics statistics);
__global__ void cudaCompactLineageStatistics(SimulationStatistics statistics);

__global__ void cudaPrepareLineageAccumulatorGC(SimulationStatistics statistics);
__global__ void cudaLineageAccumulatorGC(SimulationStatistics statistics);
__global__ void cudaFinishLineageAccumulatorGC(SimulationStatistics statistics);
