#pragma once

#include <cstdint>

#include "SimulationParametersSpotValues.h"

enum class SpotShape
{
    Circular, Rectangular
};

struct SimulationParametersSpot{
    uint32_t color = 0;
    float posX = 0;
    float posY = 0;

    float fadeoutRadius = 100.0f;

    SpotShape shape = SpotShape::Circular;

    //for Circular spots
    float coreRadius = 100.0f;
    
    //for rectangular spots
    float width = 100.0f;
    float height = 200.0f;

    SimulationParametersSpotValues values;

    bool operator==(SimulationParametersSpot const& other) const
    {
        return color == other.color && posX == other.posX && posY == other.posY && fadeoutRadius == other.fadeoutRadius && coreRadius == other.coreRadius
            && width == other.width && height == other.height && values == other.values;
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