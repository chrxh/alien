#pragma once

#include "SimulationParameters.h"
#include "GpuSettings.h"
#include "GeneralSettings.h"

struct Settings
{
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
    GpuSettings gpuSettings;
};
