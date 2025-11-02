#pragma once

#include "CudaSettings.h"
#include "SimulationParameters.h"

struct SettingsForSimulation
{
    int worldSizeX;
    int worldSizeY;
    SimulationParameters simulationParameters;
    CudaSettings cudaSettings;
};
