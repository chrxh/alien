#pragma once

#include <optional>
#include <string>
#include <variant>

#include <EngineInterface/SimulationParameters.h>

class LocationHelper
{
public:
    static LocationType getLocationType(int orderNumber, SimulationParameters const& parameters);

    static int& findOrderNumberRef(SimulationParameters& parameters, int orderNumber);
    static int findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber);

    static void decreaseOrderNumber(SimulationParameters& parameters, int orderNumber);
    static void increaseOrderNumber(SimulationParameters& parameters, int orderNumber);

    static void adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset);

    static std::string generateLayerName(SimulationParameters const& parameters);
    static std::string generateSourceName(SimulationParameters const& parameters);

    // The base location has id 0
    static int getLocationId(SimulationParameters const& parameters, int orderNumber);
    static void setLocationId(SimulationParameters& parameters, int orderNumber, int locationId);
    static std::optional<int> findOrderNumber(SimulationParameters const& parameters, int locationId);
    static int getMaxLocationId(SimulationParameters const& parameters);

    // Layers get 1..n, radiation sources the following ids
    static void assignLocationIds(SimulationParameters& parameters);
};
