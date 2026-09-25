#pragma once

#include <optional>
#include <string>
#include <variant>

#include <Base/Singleton.h>

#include "SimulationParameters.h"

class LocationEditService
{
    MAKE_SINGLETON(LocationEditService);

public:
    LocationType getLocationType(int orderNumber, SimulationParameters const& parameters) const;

    int& findOrderNumberRef(SimulationParameters& parameters, int orderNumber) const;
    int findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber) const;

    void decreaseOrderNumber(SimulationParameters& parameters, int orderNumber) const;
    void increaseOrderNumber(SimulationParameters& parameters, int orderNumber) const;

    void adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset) const;

    std::string generateLayerName(SimulationParameters const& parameters) const;
    std::string generateSourceName(SimulationParameters const& parameters) const;

    int getLocationId(SimulationParameters const& parameters, int orderNumber) const;
    void setLocationId(SimulationParameters& parameters, int orderNumber, int locationId) const;
    std::optional<int> findOrderNumber(SimulationParameters const& parameters, int locationId) const;
    int getMaxLocationId(SimulationParameters const& parameters) const;

    void assignLocationIds(SimulationParameters& parameters) const;
};
