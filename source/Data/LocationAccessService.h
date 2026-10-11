#pragma once

#include <optional>

#include <Base/Singleton.h>

#include "SimulationParameters.h"

class LocationAccessService
{
    MAKE_SINGLETON(LocationAccessService);

public:
    LocationType getLocationType(int orderNumber, SimulationParameters const& parameters) const;
    int findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber) const;

    int getLocationId(SimulationParameters const& parameters, int orderNumber) const;
    std::optional<int> findOrderNumber(SimulationParameters const& parameters, int locationId) const;
    int getMaxLocationId(SimulationParameters const& parameters) const;
};
