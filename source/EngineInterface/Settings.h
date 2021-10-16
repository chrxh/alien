#pragma once

#include "SimulationParameters.h"
#include "SimulationParametersSpots.h"
#include "GpuSettings.h"
#include "GeneralSettings.h"
#include "FlowFieldSettings.h"

struct Settings
{
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
    SimulationParametersSpots simulationParametersSpots;
    FlowFieldSettings flowFieldSettings;
};
