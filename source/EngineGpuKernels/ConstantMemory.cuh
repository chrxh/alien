#pragma once

#include "EngineInterface/FlowFieldSettings.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationParametersSpots.h"
#include "EngineInterface/GpuSettings.h"

__constant__ __device__ GpuSettings gpuConstants;
__constant__ __device__ SimulationParameters cudaSimulationParameters;
__constant__ __device__ SimulationParametersSpots cudaSimulationParametersSpots;
__constant__ __device__ FlowFieldSettings cudaFlowFieldSettings;
__constant__ __device__ int cudaImageBlurFactors[7];
