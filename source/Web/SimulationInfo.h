#pragma once

#include "Definitions.h"

struct SimulationInfo
{
    int simulationId;
    bool isActive = false;
    string simulationName;
    string userName;
    IntVector2D worldSize;
    int64_t timestep = 0;
    string description;
};
