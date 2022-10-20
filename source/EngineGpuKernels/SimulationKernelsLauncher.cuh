#pragma once

#include "EngineInterface/Settings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "SimulationKernels.cuh"

class _SimulationKernelsLauncher
{
public:
    _SimulationKernelsLauncher();

    void calcTimestep(Settings const& settings, SimulationData const& simulationData, SimulationResult const& result);

private:
    bool isRigidityUpdateEnabled(Settings const& settings) const;

    GarbageCollectorKernelsLauncher _garbageCollector;
};

