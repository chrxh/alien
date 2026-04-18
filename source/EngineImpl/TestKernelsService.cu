#include "TestKernelsService.cuh"

#include <EngineInterface/EngineConstants.h>

#include <EngineKernels/Macros.cuh>
#include <EngineKernels/TestKernels.cuh>

void TestKernelsService::init()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBoolResult);
}

void TestKernelsService::shutdown()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBoolResult);
}

void TestKernelsService::testOnly_mutate(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId)
{
    KERNEL_CALL_MOD(cudaTestMutate, NEURONS_PER_CELL, data, objectId);
}

void TestKernelsService::testOnly_createConnection(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t objectId1, uint64_t objectId2)
{
    KERNEL_CALL_1_1(cudaTestCreateConnection, data, objectId1, objectId2);
}

bool TestKernelsService::testOnly_arePointersValid(CudaSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_cudaBoolResult, true);
    KERNEL_CALL(cudaTestArePointersValid, data, _cudaBoolResult);
    return copyToHost(_cudaBoolResult);
}
