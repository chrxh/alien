#include "TestKernelsService.cuh"

#include <EngineInterface/EngineConstants.h>

#include <EngineKernels/KernelLauncher.cuh>
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

void TestKernelsService::testOnly_mutate(
    KernelLaunchSettings const& launchSettings,
    SimulationData const& data,
    SimulationStatistics const& statistics,
    uint64_t objectId)
{
    launchKernelOnDefaultStream(KERNEL(cudaTestMutate), LaunchConfig{launchSettings.numBlocks, NEURAL_NET_INPUTS}, data, statistics, objectId);
}

void TestKernelsService::testOnly_voidUnreachableNodes(KernelLaunchSettings const& launchSettings, SimulationData const& data, uint64_t objectId)
{
    launchKernelOnDefaultStream(KERNEL(cudaTestVoidUnreachableNodes), LaunchConfig{launchSettings.numBlocks, NEURAL_NET_INPUTS}, data, objectId);
}

void TestKernelsService::testOnly_removeUnusedGenes(KernelLaunchSettings const& launchSettings, SimulationData const& data, uint64_t objectId)
{
    launchKernelOnDefaultStream(KERNEL(cudaTestRemoveUnreachableGenesFromRoot), LaunchConfig{launchSettings.numBlocks, NEURAL_NET_INPUTS}, data, objectId);
}

void TestKernelsService::testOnly_removeGeneCycles(KernelLaunchSettings const& launchSettings, SimulationData const& data, uint64_t objectId)
{
    launchKernelOnDefaultStream(KERNEL(cudaTestRemoveGeneCycles), LaunchConfig{launchSettings.numBlocks, NEURAL_NET_INPUTS}, data, objectId);
}

void TestKernelsService::testOnly_limitGenesWithSeparation(KernelLaunchSettings const& launchSettings, SimulationData const& data, uint64_t objectId)
{
    launchKernelOnDefaultStream(KERNEL(cudaTestLimitGenesWithSeparation), LaunchConfig{launchSettings.numBlocks, NEURAL_NET_INPUTS}, data, objectId);
}

void TestKernelsService::testOnly_createConnection(KernelLaunchSettings const& launchSettings, SimulationData const& data, uint64_t objectId1, uint64_t objectId2)
{
    launchKernelOnDefaultStream(KERNEL(cudaTestCreateConnection), LaunchConfig{1, 1}, data, objectId1, objectId2);
}

void TestKernelsService::testOnly_createConnectionWithAbsAngle(
    KernelLaunchSettings const& launchSettings,
    SimulationData const& data,
    uint64_t objectId1,
    uint64_t objectId2,
    float desiredDistance,
    float desiredAbsAngle1,
    float desiredAbsAngle2)
{
    launchKernelOnDefaultStream(
        KERNEL(cudaTestCreateConnectionWithAbsAngle), LaunchConfig{1, 1}, data, objectId1, objectId2, desiredDistance, desiredAbsAngle1, desiredAbsAngle2);
}

bool TestKernelsService::testOnly_isDataValid(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    setValueToDevice(_cudaBoolResult, true);
    launchKernelOnDefaultStream(KERNEL(cudaTestIsDataValid), LaunchConfig{launchSettings.numBlocks, 8}, data, _cudaBoolResult);
    return copyToHost(_cudaBoolResult);
}
