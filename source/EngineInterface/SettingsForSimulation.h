#pragma once

#include "KernelLaunchSettings.h"
#include "SimulationParameters.h"

struct SettingsForSimulation
{
    int worldSizeX;
    int worldSizeY;
    SimulationParameters simulationParameters;
    KernelLaunchSettings kernelLaunchSettings;
};
