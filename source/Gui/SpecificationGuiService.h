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
        int orderNumber) const;

    void createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const;

private:
    void createWidgetsFromParameterSpecs(
        std::vector<ParameterSpec> const& parameterSpecs,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int orderNumber) const;

    void createWidgetsForBoolSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void createWidgetsForIntSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void createWidgetsForFloatSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, SimulationFacade const& simulationFacade,int orderNumber) const;
    void createWidgetsForFloat2Spec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, SimulationFacade const& simulationFacade, int orderNumber) const;
    void createWidgetsForChar64Spec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void createWidgetsForAlternativeSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, SimulationFacade const& simulationFacade,int orderNumber) const;
    void createWidgetsForColorPickerSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
    void createWidgetsForColorTransitionRulesSpec(ParameterSpec const& parameterSpec, SimulationParameters& parameters, SimulationParameters& origParameters, int orderNumber) const;
};
