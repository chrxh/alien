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

void SelectionKernelsService::removeSelection(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelection), LaunchConfig{gpuSettings.numBlocks, 8}, data, false);
}

void SelectionKernelsService::swapSelection(KernelLaunchSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelection), LaunchConfig{gpuSettings.numBlocks, 8}, data, true);
    launchKernelOnDefaultStream(KERNEL(cudaSwapSelection), LaunchConfig{gpuSettings.numBlocks, 8}, switchData.pos, switchData.radius, data);
    rolloutSelection(gpuSettings, data);
}

void SelectionKernelsService::switchSelection(KernelLaunchSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    launchKernelOnDefaultStream(KERNEL(cudaExistsSelection), LaunchConfig{gpuSettings.numBlocks, 8}, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        launchKernelOnDefaultStream(
            "cudaSetSelection",
            static_cast<void (*)(float2, float, SimulationData)>(cudaSetSelection),
            LaunchConfig{gpuSettings.numBlocks, 8},
            switchData.pos,
            switchData.radius,
            data);
        rolloutSelection(gpuSettings, data);
    }
}

void SelectionKernelsService::setSelection(KernelLaunchSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData)
{
    launchKernelOnDefaultStream(
        "cudaSetSelection", static_cast<void (*)(AreaSelectionData, SimulationData)>(cudaSetSelection), LaunchConfig{gpuSettings.numBlocks, 8}, setData, data);
    rolloutSelection(gpuSettings, data);
}

void SelectionKernelsService::updateSelection(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelection), LaunchConfig{gpuSettings.numBlocks, 8}, data, true);
    rolloutSelection(gpuSettings, data);
}

void SelectionKernelsService::rolloutSelection(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        launchKernelOnDefaultStream(KERNEL(cudaRolloutSelectionStep), LaunchConfig{gpuSettings.numBlocks, 8}, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}
