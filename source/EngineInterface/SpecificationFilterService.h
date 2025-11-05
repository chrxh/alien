#pragma once

#include <Base/Singleton.h>

#include "ParametersFilter.h"
#include "SimulationParametersSpecification.h"

class SpecificationFilterService
{
    MAKE_SINGLETON(SpecificationFilterService);

public:
    ParametersSpec filter(ParametersSpec const& spec, ParametersFilter const& filter) const;

private:
    bool matchesFilter(std::string const& name, ParametersFilter const& filter) const;
    ParameterSpec filterParameterSpec(ParameterSpec const& spec, ParametersFilter const& filter) const;
    std::vector<ParameterSpec> filterAlternativeSpecs(std::vector<ParameterSpec> const& specs, ParametersFilter const& filter) const;
};
