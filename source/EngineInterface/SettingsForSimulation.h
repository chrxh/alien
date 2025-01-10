#pragma once

#include "SimulationParameters.h"
#include "GpuSettings.h"

struct SettingsForSimulation
{
    int worldSizeX;
    int worldSizeY;
    SimulationParameters simulationParameters;
    GpuSettings gpuSettings;
};
