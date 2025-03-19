#pragma once

#include "Base/Singleton.h"

#include "SimulationParametersSpecification.h"

class SimulationParametersSpecificationService
{
    MAKE_SINGLETON(SimulationParametersSpecificationService);

public:
    ParametersSpec createParametersSpec() const;
};
