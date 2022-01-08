#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "SimulationKernels.cuh"
#include "GarbageCollectorKernelLauncher.cuh"

class SimulationKernelLauncher
{
public:
    void calcTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData, SimulationResult const& result);

private:
    GarbageCollectorKernelLauncher _garbageCollector;
};

