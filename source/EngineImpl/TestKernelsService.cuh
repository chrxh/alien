#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

#include <EngineKernels/Definitions.cuh>

class TestKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TestKernelsService);

public:
    void init();
    void shutdown();

    void testOnly_mutate(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId);
    void testOnly_createConnection(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId1, uint64_t objectId2);
    void testOnly_createConnectionWithAbsAngle(
        CudaSettings const& gpuSettings,
        SimulationData const& data,
        uint64_t objectId1,
        uint64_t objectId2,
        float desiredDistance,
        float desiredAbsAngle1,
        float desiredAbsAngle2);
    bool testOnly_isDataValid(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    TestKernelsService() = default;

    bool* _cudaBoolResult = nullptr;
};
