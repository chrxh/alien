#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SimulationParametersSpecification.h>

#include "Definitions.h"
#include "Base/Cache.h"

class SpecificationGuiService
{
    MAKE_SINGLETON(SpecificationGuiService);

public:
    void createWidgetsForParameters(
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int orderNumber,
        ParametersFilter const& filter) const;

    void createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const;

private:
    void createWidgetsForParameterGroup(
        std::vector<ParameterSpec> const& parameterSpecs,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int orderNumber,
        ParametersFilter const& filter) const;

    void createWidgetsForBoolSpec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForIntSpec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForFloatSpec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForFloat2Spec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForChar64Spec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForAlternativeSpec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        SimulationFacade const& simulationFacade,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForColorPickerSpec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        int orderNumber,
        ParametersFilter const& filter) const;
    void createWidgetsForColorTransitionRulesSpec(
        ParameterSpec const& parameterSpec,
        bool enabled,
        SimulationParameters& parameters,
        SimulationParameters& origParameters,
        int orderNumber,
        ParametersFilter const& filter) const;

private:
    Cache<ParametersFilter, ParametersSpec, 10000> _specCache;
};
