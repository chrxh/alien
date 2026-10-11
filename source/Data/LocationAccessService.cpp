#include "LocationAccessService.h"

#include <algorithm>

#include <Base/Definitions.h>

LocationType LocationAccessService::getLocationType(int orderNumber, SimulationParameters const& parameters) const
{
    if (orderNumber == 0) {
        return LocationType::Base;
    } else {
        for (int i = 0; i < parameters.numLayers; ++i) {
            if (parameters.layerOrderNumbers[i] == orderNumber) {
                return LocationType::Layer;
            }
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            if (parameters.sourceOrderNumbers[i] == orderNumber) {
                return LocationType::Source;
            }
        }
    }
    CHECK(false);
}

int LocationAccessService::findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerOrderNumbers[i] == orderNumber) {
            return i;
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceOrderNumbers[i] == orderNumber) {
            return i;
        }
    }
    CHECK(false);
}

int LocationAccessService::getLocationId(SimulationParameters const& parameters, int orderNumber) const
{
    auto locationType = getLocationType(orderNumber, parameters);
    if (locationType == LocationType::Base) {
        return 0;
    }
    auto index = findLocationArrayIndex(parameters, orderNumber);
    return locationType == LocationType::Layer ? parameters.layerIds[index] : parameters.sourceIds[index];
}

std::optional<int> LocationAccessService::findOrderNumber(SimulationParameters const& parameters, int locationId) const
{
    if (locationId == 0) {
        return 0;
    }
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerIds[i] == locationId) {
            return parameters.layerOrderNumbers[i];
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceIds[i] == locationId) {
            return parameters.sourceOrderNumbers[i];
        }
    }
    return std::nullopt;
}

int LocationAccessService::getMaxLocationId(SimulationParameters const& parameters) const
{
    auto result = 0;
    for (int i = 0; i < parameters.numLayers; ++i) {
        result = std::max(result, parameters.layerIds[i]);
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        result = std::max(result, parameters.sourceIds[i]);
    }
    return result;
}
