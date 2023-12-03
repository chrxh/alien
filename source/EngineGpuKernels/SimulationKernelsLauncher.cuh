#pragma once

#include "EngineInterface/Settings.h"
#include "EngineInterface/RawStatisticsData.h"

#include "Definitions.cuh"
#include "Macros.cuh"

class _SimulationKernelsLauncher
{
public:
    _SimulationKernelsLauncher();

    void calcTimestep(Settings const& settings, SimulationData const& simulationData, SimulationStatistics const& statistics);
    bool updateSimulationParametersAfterTimestep(
        Settings& settings,
        SimulationData const& simulationData,
        RawStatisticsData const& statistics);  //returns true if parameters have been changed
    void prepareForSimulationParametersChanges(Settings const& settings, SimulationData const& simulationData);

private:
    bool isRigidityUpdateEnabled(Settings const& settings) const;

    GarbageCollectorKernelsLauncher _garbageCollector;
    MaxAgeBalancer _maxAgeBalancer;
};

