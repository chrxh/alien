#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationParametersSpecification.h"

class ParametersSpecGuiService
{
    MAKE_SINGLETON(ParametersSpecGuiService);

public:
    enum class LocationType
    {
        Base,
        Spot,
        Source
    };
    void createWidgetsFromSpec(
        ParametersSpec const& parametersSpecs,
        LocationType locationType,
        SimulationParameters& parameters,
        SimulationParameters& origParameters) const;

private:
    bool isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const;
    bool isVisible(ParameterOrAlternativeSpec const& parameterOrAlternativeSpec, LocationType locationType) const;
};
