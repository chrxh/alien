#include "LocationHelper.h"

#include "Base/Definitions.h"

LocationType LocationHelper::getLocationType(int locationIndex, SimulationParameters const& parameters)
{
    if (locationIndex == 0) {
        return LocationType::Base;
    } else {
        for (int i = 0; i < parameters.numZones.value; ++i) {
            if (parameters.zoneLocationIndex.zoneValues[i] == locationIndex) {
                return LocationType::Zone;
            }
        }
        for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
            if (parameters.sourceLocationIndex.sourceValues[i] == locationIndex) {
                return LocationType::Source;
            }
        }
    }
    CHECK(false);
}

int& LocationHelper::findLocationIndexRef(SimulationParameters& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numZones.value; ++i) {
        if (parameters.zoneLocationIndex.zoneValues[i] == locationIndex) {
            return parameters.zoneLocationIndex.zoneValues[i];
        }
    }
    for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
        if (parameters.sourceLocationIndex.sourceValues[i] == locationIndex) {
            return parameters.sourceLocationIndex.sourceValues[i];
        }
    }

    CHECK(false);
}

int LocationHelper::findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numZones.value; ++i) {
        if (parameters.zoneLocationIndex.zoneValues[i] == locationIndex) {
            return i;
        }
    }
    for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
        if (parameters.sourceLocationIndex.sourceValues[i] == locationIndex) {
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
    for (int i = 0; i < parameters.numZones.value + parameters.numRadiationSources.value + 1; ++i) {
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
    for (int i = 0; i < parameters.numZones.value + parameters.numRadiationSources.value + 1; ++i) {
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

std::map<int, int> LocationHelper::adaptLocationIndex(SimulationParameters& parameters, int fromLocationIndex, int offset)
{
    std::map<int, int> result;
    result.emplace(0, 0);
    for (int i = 0; i < parameters.numZones.value; ++i) {
        auto& locationIndex = parameters.zoneLocationIndex.zoneValues[i];
        if (locationIndex >= fromLocationIndex) {
            result.emplace(locationIndex, locationIndex + offset);
            locationIndex += offset;
        } else {
            result.emplace(locationIndex, locationIndex);
        }
    }
    for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
        auto& locationIndex = parameters.sourceLocationIndex.sourceValues[i];
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
        for (int i = 0; i < parameters.numZones.value; ++i) {
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
        for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
            auto name = std::string(parameters.sourceName.sourceValues[i]);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}
