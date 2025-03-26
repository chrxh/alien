#pragma once

#include "Base/Singleton.h"

#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

class SimulationParametersSpecificationService
{
    MAKE_SINGLETON(SimulationParametersSpecificationService);

public:
    ParametersSpec createParametersSpec() const;

    template <typename T>
    T& getBaseParameterRef(size_t valueAddress, SimulationParameters& parameters) const;
    template <typename T>
    T& getZoneParameterRef(size_t valueAddress, SimulationParameters& parameters, int locationIndex) const;

    template <typename T>
    T& getParameterRef(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    template <typename T>
    T& getParameterRef(bool visibleInBase, bool visibleInZone, bool visibleInSource, size_t valueAddress, SimulationParameters& parameters, int locationIndex) const;

    bool& getExpertSettingsToggleRef(ParameterGroupSpec const& spec, SimulationParameters& parameters) const;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
template <typename T>
T& SimulationParametersSpecificationService::getBaseParameterRef(size_t valueAddress, SimulationParameters& parameters) const
{
    return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters) + valueAddress));
}

template <typename T>
T& SimulationParametersSpecificationService::getZoneParameterRef(size_t valueAddress, SimulationParameters& parameters, int locationIndex) const
{
    if (locationIndex == 0) {
        return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.baseValues) + valueAddress));
    }
    for (int i = 0; i < parameters.numZones; ++i) {
        if (parameters.zone[i].locationIndex == locationIndex) {
            return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.zone[i].values) + valueAddress));
        }
    }
    CHECK(false);
}

template <typename T>
T& SimulationParametersSpecificationService::getParameterRef(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    return getParameterRef<T>(spec._visibleInBase, spec._visibleInZone, spec._visibleInSource, spec._valueAddress.value(), parameters, locationIndex);
}

template <typename T>
T& SimulationParametersSpecificationService::getParameterRef(
    bool visibleInBase,
    bool visibleInZone,
    bool visibleInSource,
    size_t valueAddress,
    SimulationParameters& parameters,
    int locationIndex) const
{
    if (visibleInBase && !visibleInZone && !visibleInSource) {
        return getBaseParameterRef<T>(valueAddress, parameters);
    } else if (visibleInBase && visibleInZone && !visibleInSource) {
        return getZoneParameterRef<T>(valueAddress, parameters, locationIndex);
    }
    CHECK(false);
}
