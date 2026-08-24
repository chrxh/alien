#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/KernelLaunchSettings.h>

#include <EngineKernels/Definitions.cuh>

class TestKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TestKernelsService);

public:
    void init();
    void shutdown();

    void testOnly_mutate(KernelLaunchSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& statistics, uint64_t objectId);
    void testOnly_voidUnreachableNodes(KernelLaunchSettings const& gpuSettings, SimulationData const& data, uint64_t objectId);
    void testOnly_removeUnusedGenes(KernelLaunchSettings const& gpuSettings, SimulationData const& data, uint64_t objectId);
    void testOnly_removeGeneCycles(KernelLaunchSettings const& gpuSettings, SimulationData const& data, uint64_t objectId);
    void testOnly_limitGenesWithSeparation(KernelLaunchSettings const& gpuSettings, SimulationData const& data, uint64_t objectId);
    void testOnly_createConnection(KernelLaunchSettings const& gpuSettings, SimulationData const& data, uint64_t objectId1, uint64_t objectId2);
    void testOnly_createConnectionWithAbsAngle(
        KernelLaunchSettings const& gpuSettings,
        SimulationData const& data,
        uint64_t objectId1,
        uint64_t objectId2,
        float desiredDistance,
        float desiredAbsAngle1,
        float desiredAbsAngle2);
    bool testOnly_isDataValid(KernelLaunchSettings const& gpuSettings, SimulationData const& data);

private:
    TestKernelsService() = default;

    bool* _cudaBoolResult = nullptr;
};
