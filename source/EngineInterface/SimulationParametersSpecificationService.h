#pragma once

#include "Base/Singleton.h"

#include "SimulationParametersSpecification.h"

class SimulationParametersSpecificationService
{
    MAKE_SINGLETON(SimulationParametersSpecificationService);

public:
    ParametersSpec createParametersSpec() const;

    float& getValueRef(FloatParameterSpec const& spec, SimulationParameters& parameters) const;
    bool& getValueRef(BoolParameterSpec const& spec, SimulationParameters& parameters) const;
};
