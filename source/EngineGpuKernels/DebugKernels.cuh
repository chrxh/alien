#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CellProcessor.cuh"
#include "ParticleProcessor.cuh"

__device__ void DEBUG_checkCells(SimulationData& data, float* sumEnergy, int location);
__device__ void DEBUG_checkParticles(SimulationData& data, float* sumEnergy, int location);
__global__ void DEBUG_checkCellsAndParticles(SimulationData data, float* sumEnergy, int location);
__global__ void DEBUG_kernel(SimulationData data, int location);
