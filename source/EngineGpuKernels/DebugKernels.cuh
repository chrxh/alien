#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EnergyProcessor.cuh"
#include "SimulationData.cuh"


__global__ void DEBUG_checkAngles(SimulationData data);
__global__ void DEBUG_checkCellsAndParticles(SimulationData data, float* sumEnergy, int location);
__global__ void DEBUG_kernel(SimulationData data, int location);
