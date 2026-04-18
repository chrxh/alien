#pragma once

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/SimulationParameters.h>

__constant__ extern CudaSettings cudaSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
