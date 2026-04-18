#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

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
