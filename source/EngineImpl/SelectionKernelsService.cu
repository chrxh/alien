#include "SelectionKernelsService.cuh"

#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>
#include <EngineKernels/SelectionKernels.cuh>

#include "GarbageCollectorKernelsService.cuh"

void SelectionKernelsService::init()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _cudaRolloutResult);
    memoryManager.acquireMemory(1, _cudaSwitchResult);
}

void SelectionKernelsService::shutdown()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_cudaRolloutResult);
    memoryManager.freeMemory(_cudaSwitchResult);
}

void SelectionKernelsService::removeSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelection), LaunchConfig{launchSettings.numBlocks, 8}, data, false);
}

void SelectionKernelsService::swapSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelection), LaunchConfig{launchSettings.numBlocks, 8}, data, true);
    launchKernelOnDefaultStream(KERNEL(cudaSwapSelection), LaunchConfig{launchSettings.numBlocks, 8}, switchData.pos, switchData.radius, data);
    rolloutSelection(launchSettings, data);
}

void SelectionKernelsService::switchSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    launchKernelOnDefaultStream(KERNEL(cudaExistsSelection), LaunchConfig{launchSettings.numBlocks, 8}, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        launchKernelOnDefaultStream(KERNEL(cudaSetSelectionAtPoint), LaunchConfig{launchSettings.numBlocks, 8}, switchData.pos, switchData.radius, data);
        rolloutSelection(launchSettings, data);
    }
}

void SelectionKernelsService::setSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, AreaSelectionData const& setData)
{
    launchKernelOnDefaultStream(KERNEL(cudaSetSelectionInArea), LaunchConfig{launchSettings.numBlocks, 8}, setData, data);
    rolloutSelection(launchSettings, data);
}

void SelectionKernelsService::updateSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelection), LaunchConfig{launchSettings.numBlocks, 8}, data, true);
    rolloutSelection(launchSettings, data);
}

void SelectionKernelsService::rolloutSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        launchKernelOnDefaultStream(KERNEL(cudaRolloutSelectionStep), LaunchConfig{launchSettings.numBlocks, 8}, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}
