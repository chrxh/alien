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
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
template <typename T>
T& SimulationParametersSpecificationService::getValueRef(ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    if (spec._visibleInBase && !spec._visibleInZone && !spec._visibleInSource) {
        return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters) + spec._valueAddress));
    } else if (spec._visibleInBase && spec._visibleInZone && !spec._visibleInSource) {
        if (locationIndex == 0) {
            return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.baseValues) + spec._valueAddress));
        }
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex) {
                return *(reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.zone[i].values) + spec._valueAddress));
            }
        }
    }
    CHECK(false);
}