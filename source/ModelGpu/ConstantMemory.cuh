#pragma once

#include "ModelBasic/SimulationParameters.h"
#include "CudaConstants.h"

__constant__ __device__ CudaConstants cudaConstants;
__constant__ __device__ SimulationParameters cudaSimulationParameters;
