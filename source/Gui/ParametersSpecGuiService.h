#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationParametersSpecification.h"

class ParametersSpecGuiService
{
    MAKE_SINGLETON(ParametersSpecGuiService);

public:
    void createWidgetsForParameters(SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;

    void createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const;

private:
    void createWidgetsFromParameterSpecs(
        std::vector<ParameterSpec> const& parameterSpecs,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        int locationIndex) const;

    void createWidgetsForBoolSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForIntSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForFloatSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForChar64Spec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForAlternativeSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForColorPickerSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForColorTransitionRulesSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;

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
