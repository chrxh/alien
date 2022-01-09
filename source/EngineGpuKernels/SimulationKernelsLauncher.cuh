#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "SimulationKernels.cuh"

class _SimulationKernelsLauncher
{
public:
    _SimulationKernelsLauncher();

    void calcTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData, SimulationResult const& result);

private:
    GarbageCollectorKernelsLauncher _garbageCollector;
};

