#include "LocationEditService.h"

#include <Base/Definitions.h>

LocationType LocationEditService::getLocationType(int orderNumber, SimulationParameters const& parameters) const
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

int& LocationEditService::findOrderNumberRef(SimulationParameters& parameters, int orderNumber) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerOrderNumbers[i] == orderNumber) {
            return parameters.layerOrderNumbers[i];
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceOrderNumbers[i] == orderNumber) {
            return parameters.sourceOrderNumbers[i];
        }
    }

    CHECK(false);
}

int LocationEditService::findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber) const
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

void LocationEditService::decreaseOrderNumber(SimulationParameters& parameters, int orderNumber) const
{
    auto& orderNumberRef1 = findOrderNumberRef(parameters, orderNumber);
    auto& orderNumberRef2 = findOrderNumberRef(parameters, orderNumber - 1);
    --orderNumberRef1;
    ++orderNumberRef2;
}

void LocationEditService::increaseOrderNumber(SimulationParameters& parameters, int orderNumber) const
{
    auto& orderNumberRef1 = findOrderNumberRef(parameters, orderNumber);
    auto& orderNumberRef2 = findOrderNumberRef(parameters, orderNumber + 1);
    ++orderNumberRef1;
    --orderNumberRef2;
}

void LocationEditService::adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        auto& orderNumber = parameters.layerOrderNumbers[i];
        if (orderNumber >= fromOrderNumber) {
            orderNumber += offset;
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        auto& orderNumber = parameters.sourceOrderNumbers[i];
        if (orderNumber >= fromOrderNumber) {
            orderNumber += offset;
        }
    }
}

std::string LocationEditService::generateLayerName(SimulationParameters const& parameters) const
{
    int counter = 0;
    bool alreadyUsed;
    std::string result;
    do {
        alreadyUsed = false;
        result = "Layer " + std::to_string(counter++);
        for (int i = 0; i < parameters.numLayers; ++i) {
            auto name = std::string(parameters.layerName.layerValues[i]);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}

std::string LocationEditService::generateSourceName(SimulationParameters const& parameters) const
{
    int counter = 0;
    bool alreadyUsed;
    std::string result;
    do {
        alreadyUsed = false;
        result = "Radiation " + std::to_string(counter++);
        for (int i = 0; i < parameters.numSources; ++i) {
            auto name = std::string(parameters.sourceName.sourceValues[i]);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}

int LocationEditService::getLocationId(SimulationParameters const& parameters, int orderNumber) const
{
    auto locationType = getLocationType(orderNumber, parameters);
    if (locationType == LocationType::Base) {
        return 0;
    }
    auto index = findLocationArrayIndex(parameters, orderNumber);
    return locationType == LocationType::Layer ? parameters.layerIds[index] : parameters.sourceIds[index];
}

void LocationEditService::setLocationId(SimulationParameters& parameters, int orderNumber, int locationId) const
{
    auto locationType = getLocationType(orderNumber, parameters);
    CHECK(locationType != LocationType::Base);

    auto index = findLocationArrayIndex(parameters, orderNumber);
    if (locationType == LocationType::Layer) {
        parameters.layerIds[index] = locationId;
    } else {
        parameters.sourceIds[index] = locationId;
    }
}

std::optional<int> LocationEditService::findOrderNumber(SimulationParameters const& parameters, int locationId) const
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

int LocationEditService::getMaxLocationId(SimulationParameters const& parameters) const
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

void LocationEditService::assignLocationIds(SimulationParameters& parameters) const
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        parameters.layerIds[i] = i + 1;
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        parameters.sourceIds[i] = parameters.numLayers + i + 1;
    }
}
