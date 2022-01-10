#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"

class _GarbageCollectorKernelsLauncher
{
public:
    _GarbageCollectorKernelsLauncher();
    ~_GarbageCollectorKernelsLauncher();

    void cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& simulationData);
    void copyArrays(GpuSettings const& gpuSettings, SimulationData const& simulationData);

private:
    //gpu memory
    bool* _cudaBool;
};
