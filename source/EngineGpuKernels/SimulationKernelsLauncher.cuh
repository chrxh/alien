#pragma once

#include "EngineInterface/Settings.h"

#include "Definitions.cuh"
#include "Macros.cuh"

class _SimulationKernelsLauncher
{
public:
    _SimulationKernelsLauncher();

    void calcTimestep(Settings const& settings, SimulationData const& simulationData, SimulationStatistics const& statistics);
    bool updateSimulationParametersAfterTimestep(Settings& settings, SimulationData const& simulationData);
    void prepareForSimulationParametersChanges(Settings const& settings, SimulationData const& simulationData);

private:
    bool isRigidityUpdateEnabled(Settings const& settings) const;

    GarbageCollectorKernelsLauncher _garbageCollector;
};

