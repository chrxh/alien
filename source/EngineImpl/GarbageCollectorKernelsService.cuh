#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/KernelLaunchSettings.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/Definitions.cuh>
#include <EngineKernels/GarbageCollectorKernels.cuh>
#include <EngineKernels/Macros.cuh>

class GarbageCollectorKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GarbageCollectorKernelsService);

public:
    void init();
    void shutdown();

    void cleanupAfterTimestep(KernelLaunchSettings const& launchSettings, SimulationData const& simulationData);
    void launchCleanupForPreviewInGraph(cudaStream_t stream, int numBlocks, SimulationData const& data);
    void cleanupAfterDataManipulation(KernelLaunchSettings const& launchSettings, SimulationData const& simulationData);
    void copyArrays(KernelLaunchSettings const& launchSettings, SimulationData const& simulationData);
    void swapArrays(KernelLaunchSettings const& launchSettings, SimulationData const& simulationData);


private:
    GarbageCollectorKernelsService() = default;

    // GPU memory
    bool* _cudaBool = nullptr;
};
