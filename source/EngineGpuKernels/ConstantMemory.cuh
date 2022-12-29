#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"

__constant__ extern GpuSettings cudaThreadSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
