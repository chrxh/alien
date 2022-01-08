#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "SimulationKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"

class SimulationKernelsLauncher
{
public:
    void calcTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData, SimulationResult const& result);

private:
    GarbageCollectorKernelsLauncher _garbageCollector;
};

