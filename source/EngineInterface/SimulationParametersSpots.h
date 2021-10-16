#pragma once

#include <cstdint>

struct SimulationParametersSpot
{
    uint32_t color = 0;
    float posX = 0;
    float posY = 0;
    float coreRadius = 0;
    float fadeoutRadius = 0;

    bool operator==(SimulationParametersSpot const& other) const { return color == other.color; }
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