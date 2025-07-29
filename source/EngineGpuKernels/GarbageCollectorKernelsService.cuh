#pragma once

#include "EngineInterface/CudaSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"

class _GarbageCollectorKernelsService
{
public:
    _GarbageCollectorKernelsService();
    ~_GarbageCollectorKernelsService();

    void cleanupAfterTimestep(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterTimestepForPreview(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterDataManipulation(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void copyArrays(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void swapArrays(CudaSettings const& gpuSettings, SimulationData const& simulationData);

private:
    //gpu memory
    bool* _cudaBool;
};
