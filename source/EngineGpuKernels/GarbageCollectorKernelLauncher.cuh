#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"

class GarbageCollectorKernelLauncher
{
public:
    GarbageCollectorKernelLauncher();
    ~GarbageCollectorKernelLauncher();

    void cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& simulationData);

private:
    bool* _cudaBool;
};
