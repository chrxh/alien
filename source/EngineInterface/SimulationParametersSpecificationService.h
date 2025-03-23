#pragma once

#include "Base/Singleton.h"

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
