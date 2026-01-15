#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/MutationType.h>

#include <EngineGpuKernels/Definitions.cuh>

class TestKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TestKernelsService);

public:
    void init();
    void shutdown();

    void testOnly_mutate(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId, MutationType mutationType);
    void testOnly_mutationCheck(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId);
    void testOnly_createConnection(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId1, uint64_t objectId2);
    bool testOnly_arePointersValid(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    TestKernelsService() = default;

    bool* _cudaBoolResult = nullptr;
};
