#pragma once

#include "EngineInterface/SettingsForSimulation.h"

#include "Definitions.cuh"
#include "Macros.cuh"

class _SimulationKernelsService
{
public:
    _SimulationKernelsService();

    void calcTimestep(SettingsForSimulation const& settings, SimulationData const& simulationData, SimulationStatistics const& statistics);
    void calcTimestepForPreview(
        SettingsForSimulation const& settings,
        SimulationData const& simulationData,
        SimulationStatistics const& statistics,
        bool detailSimulation);
    void prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& simulationData);

private:
    bool isRigidityUpdateEnabled(SettingsForSimulation const& settings) const;

    GarbageCollectorKernelsService _garbageCollector;
};

