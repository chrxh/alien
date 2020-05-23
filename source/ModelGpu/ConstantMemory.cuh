#pragma once

#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/ExecutionParameters.h"
#include "CudaConstants.h"

__constant__ __device__ CudaConstants cudaConstants;
__constant__ __device__ SimulationParameters cudaSimulationParameters;
__constant__ __device__ ExecutionParameters cudaExecutionParameters;
__constant__ __device__ int cudaImageBlurFactors[7];
