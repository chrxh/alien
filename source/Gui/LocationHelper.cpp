#include "LocationHelper.h"

#include "Base/Definitions.h"

std::variant<SimulationParametersZone*, RadiationSource*> LocationHelper::findLocation(SimulationParameters& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numZones; ++i) {
        if (parameters.zone[i].locationIndex == locationIndex) {
            return &parameters.zone[i];
        }
    }
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        if (parameters.radiationSource[i].locationIndex == locationIndex) {
            return &parameters.radiationSource[i];
        }
    }
    THROW_NOT_IMPLEMENTED();
}

int LocationHelper::findLocationArrayIndex(SimulationParameters const& parameters, int locationIndex)
{
    for (int i = 0; i < parameters.numZones; ++i) {
        if (parameters.zone[i].locationIndex == locationIndex) {
            return i;
        }
    }
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        if (parameters.radiationSource[i].locationIndex == locationIndex) {
            return i;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

void LocationHelper::onDecreaseLocationIndexIntern(SimulationParameters& parameters, int locationIndex)
{
    std::variant<SimulationParametersZone*, RadiationSource*> zoneOrSource1 = findLocation(parameters, locationIndex);
    std::variant<SimulationParametersZone*, RadiationSource*> zoneOrSource2 = findLocation(parameters, locationIndex - 1);
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
}

void LocationHelper::onIncreaseLocationIndexIntern(SimulationParameters& parameters, int locationIndex)
{
    std::variant<SimulationParametersZone*, RadiationSource*> zoneOrSource1 = findLocation(parameters, locationIndex);
    std::variant<SimulationParametersZone*, RadiationSource*> zoneOrSource2 = findLocation(parameters, locationIndex + 1);
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
}

void LocationHelper::adaptLocationIndex(SimulationParameters& parameters, int fromLocationIndex, int offset)
{
    for (int i = 0; i < parameters.numZones; ++i) {
        auto& zone = parameters.zone[i];
        if (zone.locationIndex >= fromLocationIndex) {
            zone.locationIndex += offset;
        }
    }
    for (int i = 0; i < parameters.numRadiationSources; ++i) {
        auto& source = parameters.radiationSource[i];
        if (source.locationIndex >= fromLocationIndex) {
            source.locationIndex += offset;
        }
    }
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
        for (int i = 0; i < parameters.numRadiationSources; ++i) {
            auto name = std::string(parameters.radiationSource[i].name);
            if (result == name) {
                alreadyUsed = true;
                break;
            }
        }
    } while (alreadyUsed);

    return result;
}
