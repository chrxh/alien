#include "LocationHelper.h"

#include "Base/Definitions.h"

LocationType LocationHelper::getLocationType(int locationIndex, SimulationParameters const& parameters)
{
    if (locationIndex == 0) {
        return LocationType::Base;
    } else {
        for (int i = 0; i < parameters.numLayers; ++i) {
            if (parameters.layerLocationIndex[i] == locationIndex) {
                return LocationType::Layer;
            }
        }
        for (int i = 0; i < parameters.numSources; ++i) {
            if (parameters.sourceLocationIndex[i] == locationIndex) {
                return LocationType::Source;
            }
        }
    }
    CHECK(false);
}

int& LocationHelper::findLocationIndexRef(SimulationParameters& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerLocationIndex[i] == locationIndex) {
            return parameters.layerLocationIndex[i];
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceLocationIndex[i] == locationIndex) {
            return parameters.sourceLocationIndex[i];
        }
    }

    CHECK(false);
}

int LocationHelper::findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numLayers; ++i) {
        if (parameters.layerLocationIndex[i] == locationIndex) {
            return i;
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        if (parameters.sourceLocationIndex[i] == locationIndex) {
            return i;
        }
    }
    CHECK(false);
}

void LocationHelper::decreaseLocationIndex(SimulationParameters& parameters, int locationIndex)
{
    auto& locationIndexRef1 = findLocationIndexRef(parameters, locationIndex);
    auto& locationIndexRef2 = findLocationIndexRef(parameters, locationIndex - 1);
    --locationIndexRef1;
    ++locationIndexRef2;
}

void LocationHelper::increaseLocationIndex(SimulationParameters& parameters, int locationIndex)
{
    auto& locationIndexRef1 = findLocationIndexRef(parameters, locationIndex);
    auto& locationIndexRef2 = findLocationIndexRef(parameters, locationIndex + 1);
    ++locationIndexRef1;
    --locationIndexRef2;
}

std::map<int, int> LocationHelper::adaptLocationIndices(SimulationParameters& parameters, int fromLocationIndex, int offset)
{
    std::map<int, int> result;
    result.emplace(0, 0);
    for (int i = 0; i < parameters.numLayers; ++i) {
        auto& locationIndex = parameters.layerLocationIndex[i];
        if (locationIndex >= fromLocationIndex) {
            result.emplace(locationIndex, locationIndex + offset);
            locationIndex += offset;
        } else {
            result.emplace(locationIndex, locationIndex);
        }
    }
    for (int i = 0; i < parameters.numSources; ++i) {
        auto& locationIndex = parameters.sourceLocationIndex[i];
        if (locationIndex >= fromLocationIndex) {
            result.emplace(locationIndex, locationIndex + offset);
            locationIndex += offset;
        } else {
            result.emplace(locationIndex, locationIndex);
        }
    }
    return result;
}

std::string LocationHelper::generateLayerName(SimulationParameters const& parameters)
{
    int counter = 0;
    bool alreadyUsed;
    std::string result;
    do {
        alreadyUsed = false;
        result = "Layer " + std::to_string(++counter);
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
        result = "Radiation " + std::to_string(++counter);
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
