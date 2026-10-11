#pragma once

#include <Base/Singleton.h>

#include <Engine/Interface/KernelLaunchSettings.h>

#include <Engine/Impl/Kernels/Base.cuh>
#include <Engine/Impl/Kernels/Definitions.cuh>
#include <Engine/Impl/Kernels/GarbageCollectorKernels.cuh>
#include <Engine/Impl/Kernels/Macros.cuh>

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
