#pragma once

#include "EngineInterface/Settings.h"

#include "Definitions.cuh"
#include "Macros.cuh"

class _SimulationKernelsLauncher
{
public:
    _SimulationKernelsLauncher();

    bool calcSimulationParametersForNextTimestep(Settings& settings);
    void calcTimestep(Settings const& settings, SimulationData const& simulationData, SimulationStatistics const& statistics);
    void prepareForSimulationParametersChanges(Settings const& settings, SimulationData const& simulationData);

private:
    bool isRigidityUpdateEnabled(Settings const& settings) const;

    GarbageCollectorKernelsLauncher _garbageCollector;
};

