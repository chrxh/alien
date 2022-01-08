#pragma once

#include "EngineInterface/FlowFieldSettings.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationParametersSpots.h"
#include "EngineInterface/GpuSettings.h"

__constant__ __device__ static GpuSettings cudaThreadSettings;
__constant__ __device__ static SimulationParameters cudaSimulationParameters;
__constant__ __device__ static SimulationParametersSpots cudaSimulationParametersSpots;
__constant__ __device__ static FlowFieldSettings cudaFlowFieldSettings;
__constant__ __device__ static int cudaImageBlurFactors[7];
