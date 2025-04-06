#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationParametersSpecification.h"

class SpecificationGuiService
{
    MAKE_SINGLETON(SpecificationGuiService);

public:
    void createWidgetsForParameters(
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int locationIndex) const;

    void createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const;

private:
    void createWidgetsFromParameterSpecs(
        std::vector<ParameterSpec> const& parameterSpecs,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int locationIndex) const;

    void createWidgetsForBoolSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForIntSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForFloatSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, SimulationFacade const& simulationFacade,int locationIndex) const;
    void createWidgetsForFloat2Spec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, SimulationFacade const& simulationFacade, int locationIndex) const;
    void createWidgetsForChar64Spec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForAlternativeSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, SimulationFacade const& simulationFacade,int locationIndex) const;
    void createWidgetsForColorPickerSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
    void createWidgetsForColorTransitionRulesSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const;
};
