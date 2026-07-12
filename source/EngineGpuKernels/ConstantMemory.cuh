#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"

__constant__ extern GpuSettings cudaThreadSettings;
__device__ extern char cudaSimulationParametersData[sizeof(SimulationParameters)];
#define cudaSimulationParameters (*reinterpret_cast<SimulationParameters*>(cudaSimulationParametersData))
