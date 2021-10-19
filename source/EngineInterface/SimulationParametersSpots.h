#pragma once

#include <cstdint>

#include "SimulationParametersSpotValues.h"

struct SimulationParametersSpot
{
    uint32_t color = 0;
    float posX = 0;
    float posY = 0;
    float coreRadius = 0;
    float fadeoutRadius = 0;

    SimulationParametersSpotValues values;

    bool operator==(SimulationParametersSpot const& other) const
    {
        return color == other.color && posX == other.posX && posY == other.posY && coreRadius == other.coreRadius
            && fadeoutRadius == other.fadeoutRadius && values == other.values;
    }
};

struct SimulationParametersSpots
{
    int numSpots = 0;
    SimulationParametersSpot spots[2];

    bool operator==(SimulationParametersSpots const& other) const
    {
        return numSpots == other.numSpots && spots == other.spots;
    }
    bool operator!=(SimulationParametersSpots const& other) const { return !operator==(other); }
};