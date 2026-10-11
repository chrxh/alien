#pragma once

#include <Data/SimulationParameters.h>

#include "KernelLaunchSettings.h"

struct SettingsForSimulation
{
    int worldSizeX;
    int worldSizeY;
    SimulationParameters simulationParameters;
    KernelLaunchSettings kernelLaunchSettings;
};
