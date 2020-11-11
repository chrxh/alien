#pragma once

#include "Definitions.h"

struct SimulationInfo
{
    string simulationId;
    bool isActive = false;
    string simulationName;
    string userName;
    IntVector2D worldSize;
    int64_t timestep = 0;
    string description;
};
