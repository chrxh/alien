#pragma once

#include <map>
#include <string>
#include <variant>

#include "EngineInterface/SimulationParameters.h"

class LocationHelper
{
public:
    static std::variant<SimulationParametersZone*, RadiationSource*> findLocation(SimulationParameters& parameters, int locationIndex);
    static int findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex);

    // returns new by old location index
    static std::map<int, int> onDecreaseLocationIndex(SimulationParameters& parameters, int locationIndex);

    // returns new by old location index
    static std::map<int, int> onIncreaseLocationIndex(SimulationParameters& parameters, int locationIndex);

    // returns new by old location index
    static std::map<int, int> adaptLocationIndex(SimulationParameters& parameters, int fromLocationIndex, int offset);

    static std::string generateZoneName(SimulationParameters& parameters);
    static std::string generateSourceName(SimulationParameters& parameters);
};
