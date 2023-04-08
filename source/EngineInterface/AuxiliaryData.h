#pragma once

#include "Base/Definitions.h"
#include "Settings.h"

struct AuxiliaryData
{
    uint64_t timestep = 0;
    float zoom = 0;
    RealVector2D center;
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
};
