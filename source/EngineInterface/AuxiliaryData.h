#pragma once

#include <chrono>

#include "Base/Definitions.h"
#include "Settings.h"

struct AuxiliaryData
{
    uint64_t timestep = 0;
    std::chrono::milliseconds realTime;
    float zoom = 0;
    RealVector2D center;
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
};
