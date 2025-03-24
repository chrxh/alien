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

    template<typename T>
    T& getValueRef(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const;

    template <typename T>
    T& getValueRef(bool visibleInBase, bool visibleInZone, bool visibleInSource, size_t valueAddress, SimulationParameters& parameters, int locationIndex) const;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
template <typename T>
T& SimulationParametersSpecificationService::getValueRef(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    return getValueRef<T>(spec._visibleInBase, spec._visibleInZone, spec._visibleInSource, spec._valueAddress.value(), parameters, locationIndex);
}

template <typename T>
T& SimulationParametersSpecificationService::getValueRef(
    bool visibleInBase,
    bool visibleInZone,
    bool visibleInSource,
    size_t valueAddress,
    SimulationParameters& parameters,
    int locationIndex) const
{
    if (visibleInBase && !visibleInZone && !visibleInSource) {
        return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters) + valueAddress));
    } else if (visibleInBase && visibleInZone && !visibleInSource) {
        if (locationIndex == 0) {
            return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.baseValues) + valueAddress));
        }
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex) {
                return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.zone[i].values) + valueAddress));
            }
        }
    }
    CHECK(false);
}
