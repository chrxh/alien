#pragma once

#include "Base/Singleton.h"

#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

class SpecificationService
{
    MAKE_SINGLETON(SpecificationService);

public:
    ParametersSpec const& getSpec();

private:
    void createSpec();
    std::optional<ParametersSpec> _parametersSpec;
};
