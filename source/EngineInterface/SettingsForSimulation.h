#pragma once

#include "SimulationParameters.h"
#include "CudaSettings.h"

struct SettingsForSimulation
{
    int worldSizeX;
    int worldSizeY;
    SimulationParameters simulationParameters;
    CudaSettings cudaSettings;
};
