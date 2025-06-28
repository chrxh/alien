#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/CudaSettings.h"

__constant__ extern CudaSettings cudaSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
