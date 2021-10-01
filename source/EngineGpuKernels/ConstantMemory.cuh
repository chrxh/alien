#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuConstants.h"

__constant__ __device__ GpuConstants gpuConstants;
__constant__ __device__ SimulationParameters cudaSimulationParameters;
__constant__ __device__ int cudaImageBlurFactors[7];
