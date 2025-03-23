#pragma once

#include "Base/Singleton.h"

#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

class SimulationParametersSpecificationService
{
    MAKE_SINGLETON(SimulationParametersSpecificationService);

public:
    ParametersSpec createParametersSpec() const;

    float& getValueRef(FloatParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool& getValueRef(BoolParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    Char64& getValueRef(Char64ParameterSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
};
