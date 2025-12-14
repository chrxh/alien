#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SettingsForSimulation.h>

#include "Definitions.cuh"
#include "Macros.cuh"

class SimulationKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationKernelsService);

public:
    void init();
    void shutdown();

    void calcTimestep(SettingsForSimulation const& settings, SimulationData const& simulationData, SimulationStatistics const& statistics);
    void calcTimestepForPreview(
        SettingsForSimulation const& settings,
        SimulationData const& simulationData,
        SimulationStatistics const& statistics,
        bool detailSimulation);
    void prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& simulationData);

private:
    SimulationKernelsService() = default;

    bool isRigidityUpdateEnabled(SettingsForSimulation const& settings) const;
};
