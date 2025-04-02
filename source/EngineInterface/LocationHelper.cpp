#include "LocationHelper.h"

#include "Base/Definitions.h"

LocationType LocationHelper::getLocationType(int locationIndex, SimulationParameters const& parameters)
{
    if (locationIndex == 0) {
        return LocationType::Base;
    } else {
        for (int i = 0; i < parameters.numZones.value; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex) {
                return LocationType::Zone;
            }
        }
        for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
            if (parameters.radiationSource[i].locationIndex == locationIndex) {
                return LocationType::Source;
            }
        }
    }
    CHECK(false);
}

std::variant<SimulationParameters*, SimulationParametersZone*, RadiationSource*> LocationHelper::findLocation(
    SimulationParameters& parameters,
    int locationIndex)
{
    if (locationIndex == 0) {
        return &parameters;
    }
    for (int i = 0; i < parameters.numZones.value; ++i) {
        if (parameters.zone[i].locationIndex == locationIndex) {
            return &parameters.zone[i];
        }
    }
    for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
        if (parameters.radiationSource[i].locationIndex == locationIndex) {
            return &parameters.radiationSource[i];
        }
    }
    CHECK(false);
}

int LocationHelper::findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numZones.value; ++i) {
        if (parameters.zone[i].locationIndex == locationIndex) {
            return i;
        }
    }
    for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
        if (parameters.radiationSource[i].locationIndex == locationIndex) {
            return i;
        }
    }
    CHECK(false);
}

std::map<int, int> LocationHelper::onDecreaseLocationIndex(SimulationParameters& parameters, int locationIndex)
{
    auto zoneOrSource1 = findLocation(parameters, locationIndex);
    auto zoneOrSource2 = findLocation(parameters, locationIndex - 1);
    if (std::holds_alternative<SimulationParametersZone*>(zoneOrSource1)) {
        std::get<SimulationParametersZone*>(zoneOrSource1)->locationIndex -= 1;
    } else {
        std::get<RadiationSource*>(zoneOrSource1)->locationIndex -= 1;
    }
    if (std::holds_alternative<SimulationParametersZone*>(zoneOrSource2)) {
        std::get<SimulationParametersZone*>(zoneOrSource2)->locationIndex += 1;
    } else {
        std::get<RadiationSource*>(zoneOrSource2)->locationIndex += 1;
    }

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
    auto zoneOrSource1 = findLocation(parameters, locationIndex);
    auto zoneOrSource2 = findLocation(parameters, locationIndex + 1);
    if (std::holds_alternative<SimulationParametersZone*>(zoneOrSource1)) {
        std::get<SimulationParametersZone*>(zoneOrSource1)->locationIndex += 1;
    } else {
        std::get<RadiationSource*>(zoneOrSource1)->locationIndex += 1;
    }
    if (std::holds_alternative<SimulationParametersZone*>(zoneOrSource2)) {
        std::get<SimulationParametersZone*>(zoneOrSource2)->locationIndex -= 1;
    } else {
        std::get<RadiationSource*>(zoneOrSource2)->locationIndex -= 1;
    }

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
        auto& zone = parameters.zone[i];
        if (zone.locationIndex >= fromLocationIndex) {
            result.emplace(zone.locationIndex, zone.locationIndex + offset);
            zone.locationIndex += offset;
        } else {
            result.emplace(zone.locationIndex, zone.locationIndex);
        }
    }
    for (int i = 0; i < parameters.numRadiationSources.value; ++i) {
        auto& source = parameters.radiationSource[i];
        if (source.locationIndex >= fromLocationIndex) {
            result.emplace(source.locationIndex, source.locationIndex + offset);
            source.locationIndex += offset;
        } else {
            result.emplace(source.locationIndex, source.locationIndex);
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
            auto name = std::string(parameters.zone[i].name);
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
            auto name = std::string(parameters.radiationSource[i].name);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}
