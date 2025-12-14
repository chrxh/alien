#include "SelectionKernelsService.cuh"

#include "DataAccessKernels.cuh"
#include "SelectionKernels.cuh"
#include "GarbageCollectorKernelsService.cuh"

_SelectionKernelsService::_SelectionKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _cudaRolloutResult);
    memoryManager.acquireMemory(1, _cudaSwitchResult);
    memoryManager.acquireMemory(1, _cudaMinCellPosYAndIndex);
    memoryManager.acquireMemory(1, _cudaMinCellPosYAndIndex);
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsService>();
}

_SelectionKernelsService::~_SelectionKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_cudaRolloutResult);
    memoryManager.freeMemory(_cudaSwitchResult);
    memoryManager.freeMemory(_cudaMinCellPosYAndIndex);
}

void _SelectionKernelsService::removeSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, false);
}

void _SelectionKernelsService::swapSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    KERNEL_CALL(cudaSwapSelection, switchData.pos, switchData.radius, data);
    rolloutSelection(gpuSettings, data);
}

void _SelectionKernelsService::switchSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    KERNEL_CALL(cudaExistsSelection, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        KERNEL_CALL(cudaSetSelection, switchData.pos, switchData.radius, data);
        rolloutSelection(gpuSettings, data);
    }
}

void _SelectionKernelsService::setSelection(CudaSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData)
{
    KERNEL_CALL(cudaSetSelection, setData, data);
    rolloutSelection(gpuSettings, data);
}

void _SelectionKernelsService::updateSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    rolloutSelection(gpuSettings, data);
}

void _SelectionKernelsService::getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult)
{
    KERNEL_CALL_1_1(cudaResetSelectionResult, selectionResult);
    setValueToDevice(_cudaMinCellPosYAndIndex, 0xffffffffffffffffull);
    KERNEL_CALL(cudaCalcCellWithMinimalPosY, data, _cudaMinCellPosYAndIndex);
    cudaDeviceSynchronize();
    auto refCellIndex = static_cast<int>(copyToHost(_cudaMinCellPosYAndIndex) & 0xffffffff);
    KERNEL_CALL(cudaGetSelectionShallowData_step1, data);
    KERNEL_CALL(cudaGetSelectionShallowData_step2, data, refCellIndex, selectionResult);
    KERNEL_CALL_1_1(cudaFinalizeSelectionResult, selectionResult, data.cellMap);
}

void _SelectionKernelsService::rolloutSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        KERNEL_CALL(cudaRolloutSelectionStep, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}
