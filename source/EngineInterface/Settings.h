#pragma once

#include "SimulationParameters.h"
#include "GpuSettings.h"
#include "GeneralSettings.h"
#include "FlowFieldSettings.h"

struct Settings
{
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
    FlowFieldSettings flowFieldSettings;
};
