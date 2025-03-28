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
    T* getValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getPinnedValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getEnabledValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getExpertSettingsToggleRef(ParameterGroupSpec const& spec, SimulationParameters& parameters) const;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename T>
T* SimulationParametersSpecificationService::getValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<BaseValueSpec>(spec)) {
        auto baseValueSpec = std::get<BaseValueSpec>(spec);
        if (baseValueSpec._valueAddress.has_value()) {
            return reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters) + baseValueSpec._valueAddress.value());
        }
    } else if (std::holds_alternative<BaseZoneValueSpec>(spec)) {
        auto baseZoneValueSpec = std::get<BaseZoneValueSpec>(spec);

        if (locationIndex == 0 && baseZoneValueSpec._valueAddress.has_value()) {
            return reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.baseValues) + baseZoneValueSpec._valueAddress.value());
        }
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex && baseZoneValueSpec._valueAddress.has_value()) {
                return reinterpret_cast<T*>(reinterpret_cast<char*>(&parameters.zone[i].values) + baseZoneValueSpec._valueAddress.value());
            }
        }
    }
    return nullptr;
}
