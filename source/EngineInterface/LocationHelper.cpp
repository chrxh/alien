#include "LocationHelper.h"

#include "Base/Definitions.h"

LocationType LocationHelper::getLocationType(int locationIndex, SimulationParameters const& parameters)
{
    if (locationIndex == 0) {
        return LocationType::Base;
    } else {
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zoneLocationIndex[i] == locationIndex) {
                return LocationType::Zone;
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
    for (int i = 0; i < parameters.numZones; ++i) {
        if (parameters.zoneLocationIndex[i] == locationIndex) {
            return parameters.zoneLocationIndex[i];
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
    for (int i = 0; i < parameters.numZones; ++i) {
        if (parameters.zoneLocationIndex[i] == locationIndex) {
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

std::map<int, int> LocationHelper::onDecreaseLocationIndex(SimulationParameters& parameters, int locationIndex)
{
    auto& locationIndexRef1 = findLocationIndexRef(parameters, locationIndex);
    auto& locationIndexRef2 = findLocationIndexRef(parameters, locationIndex - 1);
    --locationIndexRef1;
    ++locationIndexRef2;

    std::map<int, int> result;
    for (int i = 0; i < parameters.numZones + parameters.numSources + 1; ++i) {
        if (i == locationIndex) {
            result.emplace(i, i - 1);
        } else if (i == locationIndex - 1) {
            result.emplace(i, i + 1);
        } else {
            result.emplace(i, i);
        }
    }
    return result;
}

std::map<int, int> LocationHelper::onIncreaseLocationIndex(SimulationParameters& parameters, int locationIndex)
{
    auto& locationIndexRef1 = findLocationIndexRef(parameters, locationIndex);
    auto& locationIndexRef2 = findLocationIndexRef(parameters, locationIndex + 1);
    ++locationIndexRef1;
    --locationIndexRef2;

    std::map<int, int> result;
    for (int i = 0; i < parameters.numZones + parameters.numSources + 1; ++i) {
        if (i == locationIndex) {
            result.emplace(i, i + 1);
        } else if (i == locationIndex + 1) {
            result.emplace(i, i - 1);
        } else {
            result.emplace(i, i);
        }
    }
    return result;
}

std::map<int, int> LocationHelper::adaptLocationIndices(SimulationParameters& parameters, int fromLocationIndex, int offset)
{
    std::map<int, int> result;
    result.emplace(0, 0);
    for (int i = 0; i < parameters.numZones; ++i) {
        auto& locationIndex = parameters.zoneLocationIndex[i];
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

std::string LocationHelper::generateZoneName(SimulationParameters& parameters)
{
    int counter = 0;
    bool alreadyUsed;
    std::string result;
    do {
        alreadyUsed = false;
        result = "Zone " + std::to_string(++counter);
        for (int i = 0; i < parameters.numZones; ++i) {
            auto name = std::string(parameters.zoneName.zoneValues[i]);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}

std::string LocationHelper::generateSourceName(SimulationParameters& parameters)
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
