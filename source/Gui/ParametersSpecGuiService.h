#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationParametersSpecification.h"

class ParametersSpecGuiService
{
    MAKE_SINGLETON(ParametersSpecGuiService);

public:
    void createWidgetsFromSpec(
        ParametersSpec const& parametersSpecs,
        int locationIndex,
        SimulationParameters& parameters,
        SimulationParameters& origParameters) const;

private:
    enum class LocationType
    {
        Base,
        Zone,
        Source
    };
    LocationType getLocationType(int locationIndex, SimulationParameters const& parameters) const;

    bool isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const;
    bool isVisible(ParameterSpec const& parameterSpec, LocationType locationType) const;
};
