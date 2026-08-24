#pragma once

#include <EngineInterface/KernelLaunchSettings.h>
#include <EngineInterface/SimulationParameters.h>

__constant__ extern KernelLaunchSettings kernelLaunchSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
