#pragma once

#include <map>
#include <string>
#include <variant>

#include "EngineInterface/SimulationParameters.h"

class LocationHelper
{
public:
    static LocationType getLocationType(int locationIndex, SimulationParameters const& parameters);

    static int& findLocationIndexRef(SimulationParameters& parameters, int locationIndex);
    static int findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex);

    static void decreaseLocationIndex(SimulationParameters& parameters, int locationIndex);
    static void increaseLocationIndex(SimulationParameters& parameters, int locationIndex);

    // returns new by old location index
    static std::map<int, int> adaptLocationIndices(SimulationParameters& parameters, int fromLocationIndex, int offset);

    static std::string generateLayerName(SimulationParameters const& parameters);
    static std::string generateSourceName(SimulationParameters const& parameters);
};
