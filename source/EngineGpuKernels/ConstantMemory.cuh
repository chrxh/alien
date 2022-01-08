#pragma once

#include "EngineInterface/FlowFieldSettings.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationParametersSpots.h"
#include "EngineInterface/GpuSettings.h"

__constant__ extern GpuSettings cudaThreadSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
__constant__ extern SimulationParametersSpots cudaSimulationParametersSpots;
__constant__ extern FlowFieldSettings cudaFlowFieldSettings;
