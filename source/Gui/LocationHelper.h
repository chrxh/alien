#pragma once

#include <string>
#include <variant>

#include "EngineInterface/SimulationParameters.h"

class LocationHelper
{
public:
    static std::variant<SimulationParametersZone*, RadiationSource*> findLocation(SimulationParameters& parameters, int locationIndex);
    static int findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex);

    static void onDecreaseLocationIndexIntern(SimulationParameters& parameters, int locationIndex);
    static void onIncreaseLocationIndexIntern(SimulationParameters& parameters, int locationIndex);

    static void adaptLocationIndex(SimulationParameters& parameters, int fromLocationIndex, int offset);

    static std::string generateZoneName(SimulationParameters& parameters);
    static std::string generateSourceName(SimulationParameters& parameters);
};
