#include "SelectionKernelsService.cuh"

#include <EngineGpuKernels/DataAccessKernels.cuh>
#include <EngineGpuKernels/SelectionKernels.cuh>
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

void SelectionKernelsService::removeSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, false);
}

void SelectionKernelsService::swapSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    KERNEL_CALL(cudaSwapSelection, switchData.pos, switchData.radius, data);
    rolloutSelection(gpuSettings, data);
}

void SelectionKernelsService::switchSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    KERNEL_CALL(cudaExistsSelection, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        KERNEL_CALL(cudaSetSelection, switchData.pos, switchData.radius, data);
        rolloutSelection(gpuSettings, data);
    }
}

void SelectionKernelsService::setSelection(CudaSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData)
{
    KERNEL_CALL(cudaSetSelection, setData, data);
    rolloutSelection(gpuSettings, data);
}

void SelectionKernelsService::updateSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    rolloutSelection(gpuSettings, data);
}

void SelectionKernelsService::rolloutSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        KERNEL_CALL(cudaRolloutSelectionStep, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}
