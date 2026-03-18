#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

#include <EngineGpuKernels/Base.cuh>
#include <EngineGpuKernels/Definitions.cuh>
#include <EngineGpuKernels/GarbageCollectorKernels.cuh>
#include <EngineGpuKernels/Macros.cuh>

class GarbageCollectorKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GarbageCollectorKernelsService);

public:
    void init();
    void shutdown();

    void cleanupAfterTimestep(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void launchCleanupForPreviewInGraph(cudaStream_t stream, int numBlocks, SimulationData const& data);
    void cleanupAfterDataManipulation(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void copyArrays(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void swapArrays(CudaSettings const& gpuSettings, SimulationData const& simulationData);

    
private:
    GarbageCollectorKernelsService() = default;

    //gpu memory
    bool* _cudaBool = nullptr;
};
