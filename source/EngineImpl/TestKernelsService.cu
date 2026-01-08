#include <EngineGpuKernels/Macros.cuh>
#include <EngineGpuKernels/TestKernels.cuh>

#include "TestKernelsService.cuh"

void TestKernelsService::init()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBoolResult);
}

void TestKernelsService::shutdown()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBoolResult);
}

void TestKernelsService::testOnly_mutate(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t cellId, MutationType mutationType)
{
    KERNEL_CALL(cudaTestMutate, data, cellId, mutationType);
}

void TestKernelsService::testOnly_mutationCheck(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t cellId)
{
    KERNEL_CALL(cudaTestMutationCheck, data, cellId);
}

void TestKernelsService::testOnly_createConnection(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t cellId1, uint64_t cellId2)
{
    KERNEL_CALL_1_1(cudaTestCreateConnection, data, cellId1, cellId2);
}

bool TestKernelsService::testOnly_arePointersValid(CudaSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_cudaBoolResult, true);
    KERNEL_CALL(cudaTestArePointersValid, data, _cudaBoolResult);
    return copyToHost(_cudaBoolResult);
}
