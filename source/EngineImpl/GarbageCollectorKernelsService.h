#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

struct SimulationData;

class GarbageCollectorKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GarbageCollectorKernelsService);

public:
    void init();
    void shutdown();

    void cleanupAfterTimestep(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterTimestepForPreview(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterDataManipulation(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void copyArrays(CudaSettings const& gpuSettings, SimulationData const& simulationData);
    void swapArrays(CudaSettings const& gpuSettings, SimulationData const& simulationData);

private:
    GarbageCollectorKernelsService() = default;

    //gpu memory
    bool* _cudaBool = nullptr;
};
