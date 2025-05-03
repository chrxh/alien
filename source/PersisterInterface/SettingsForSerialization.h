#pragma once

#include <chrono>

#include "Base/Definitions.h"
#include "EngineInterface/SettingsForSimulation.h"

struct SettingsForSerialization
{
    uint64_t timestep = 0;
    std::chrono::milliseconds realTime = std::chrono::milliseconds(0);
    float zoom = 4.0f;
    RealVector2D center = RealVector2D(500.0f, 500.0f);
    IntVector2D worldSize = IntVector2D(1000, 1000);
    SimulationParameters simulationParameters;
};
