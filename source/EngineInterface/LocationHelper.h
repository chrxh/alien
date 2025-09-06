#pragma once

#include <map>
#include <string>
#include <variant>

#include "EngineInterface/SimulationParameters.h"

class LocationHelper
{
public:
    static LocationType getLocationType(int orderNumber, SimulationParameters const& parameters);

    static int& findOrderNumberRef(SimulationParameters& parameters, int orderNumber);
    static int findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber);

    static void decreaseOrderNumber(SimulationParameters& parameters, int orderNumber);
    static void increaseOrderNumber(SimulationParameters& parameters, int orderNumber);

    // returns new by old location index
    static std::map<int, int> adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset);

    static std::string generateLayerName(SimulationParameters const& parameters);
    static std::string generateSourceName(SimulationParameters const& parameters);
};
