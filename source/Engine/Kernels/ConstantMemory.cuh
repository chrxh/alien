#pragma once

#include <Data/Interface/SimulationParameters.h>

#include <Engine/Interface/KernelLaunchSettings.h>

__constant__ extern KernelLaunchSettings kernelLaunchSettings;
__constant__ extern SimulationParameters cudaSimulationParameters;
