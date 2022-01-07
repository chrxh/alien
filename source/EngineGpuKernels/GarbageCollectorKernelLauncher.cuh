#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "GarbageCollectorKernels.cuh"

class GarbageCollectorKernelLauncher
{
public:
    static void cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData);
};

/**
 * Implementation
 */
void GarbageCollectorKernelLauncher::cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL(cleanupCellMap, simulationData);
    KERNEL_CALL(cleanupParticleMap, simulationData);

    KERNEL_CALL_1_1(cleanupAfterSimulationKernel, simulationData);
}
