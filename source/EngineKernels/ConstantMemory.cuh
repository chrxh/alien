#pragma once

#include <Data/SimulationParameters.h>

#include <EngineInterface/KernelLaunchSettings.h>

__constant__ extern KernelLaunchSettings kernelLaunchSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
