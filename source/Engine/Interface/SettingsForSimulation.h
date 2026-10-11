#pragma once

#include <Data/Interface/SimulationParameters.h>

#include "KernelLaunchSettings.h"

struct SettingsForSimulation
{
    int worldSizeX;
    int worldSizeY;
    SimulationParameters simulationParameters;
    KernelLaunchSettings kernelLaunchSettings;
};
