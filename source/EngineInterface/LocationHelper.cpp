#include "LocationHelper.h"

#include <Base/Definitions.h>

LocationType LocationHelper::getLocationType(int orderNumber, SimulationParameters const& parameters)
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

int& LocationHelper::findOrderNumberRef(SimulationParameters& parameters, int orderNumber)
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

int LocationHelper::findLocationArrayIndex(SimulationParameters const& parameters, int orderNumber)
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

void LocationHelper::decreaseOrderNumber(SimulationParameters& parameters, int orderNumber)
{
    auto& orderNumberRef1 = findOrderNumberRef(parameters, orderNumber);
    auto& orderNumberRef2 = findOrderNumberRef(parameters, orderNumber - 1);
    --orderNumberRef1;
    ++orderNumberRef2;
}

void LocationHelper::increaseOrderNumber(SimulationParameters& parameters, int orderNumber)
{
    auto& orderNumberRef1 = findOrderNumberRef(parameters, orderNumber);
    auto& orderNumberRef2 = findOrderNumberRef(parameters, orderNumber + 1);
    ++orderNumberRef1;
    --orderNumberRef2;
}

void LocationHelper::adaptLocationIndices(SimulationParameters& parameters, int fromOrderNumber, int offset)
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

std::string LocationHelper::generateLayerName(SimulationParameters const& parameters)
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

std::string LocationHelper::generateSourceName(SimulationParameters const& parameters)
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

int LocationHelper::getLocationId(SimulationParameters const& parameters, int orderNumber)
{
    auto locationType = getLocationType(orderNumber, parameters);
    if (locationType == LocationType::Base) {
        return 0;
    }
    auto index = findLocationArrayIndex(parameters, orderNumber);
    return locationType == LocationType::Layer ? parameters.layerIds[index] : parameters.sourceIds[index];
}

void LocationHelper::setLocationId(SimulationParameters& parameters, int orderNumber, int locationId)
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

std::optional<int> LocationHelper::findOrderNumber(SimulationParameters const& parameters, int locationId)
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

int LocationHelper::getMaxLocationId(SimulationParameters const& parameters)
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

void LocationHelper::assignLocationIds(SimulationParameters& parameters)
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        parameters.layerIds[i] = i + 1;
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        parameters.sourceIds[i] = parameters.numLayers + i + 1;
    }
}
