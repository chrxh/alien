#pragma once

#include <chrono>

#include "Base/Definitions.h"
#include "EngineInterface/SettingsForSimulation.h"

struct SettingsForSerialization
{
    uint64_t timestep = 0;
    std::chrono::milliseconds realTime;
    float zoom = 0;
    RealVector2D center;
    IntVector2D worldSize;
    SimulationParameters simulationParameters;
};
