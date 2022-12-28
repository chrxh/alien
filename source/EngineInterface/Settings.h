#pragma once

#include "SimulationParameters.h"
#include "SimulationParametersSpots.h"
#include "GpuSettings.h"
#include "GeneralSettings.h"

struct Settings
{
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
    SimulationParametersSpots simulationParametersSpots;
    GpuSettings gpuSettings;
};
