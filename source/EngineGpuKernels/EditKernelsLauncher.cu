#include "EditKernelsLauncher.cuh"

#include "EditKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"

_EditKernelsLauncher::_EditKernelsLauncher()
{
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaRolloutResult);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaSwitchResult);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaUpdateResult);
    CudaMemoryManager::getInstance().acquireMemory<float2>(1, _cudaCenter);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaNumEntities);
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsLauncher>();
}

_EditKernelsLauncher::~_EditKernelsLauncher()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaRolloutResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaSwitchResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaUpdateResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaCenter);
    CudaMemoryManager::getInstance().freeMemory(_cudaNumEntities);
}

void _EditKernelsLauncher::removeSelection(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, false);
}

void _EditKernelsLauncher::swapSelection(GpuSettings const& gpuSettings, SimulationData const& data, PointSelectionData switchData)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    KERNEL_CALL(cudaSwapSelection, switchData.pos, switchData.radius, data);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsLauncher::switchSelection(GpuSettings const& gpuSettings, SimulationData data, PointSelectionData switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    KERNEL_CALL(cudaExistsSelection, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        KERNEL_CALL(cudaSetSelection, switchData.pos, switchData.radius, data);
        rolloutSelection(gpuSettings, data);
    }
}

void _EditKernelsLauncher::setSelection(GpuSettings const& gpuSettings, SimulationData data, AreaSelectionData setData)
{
    KERNEL_CALL(cudaSetSelection, setData, data);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsLauncher::updateSelection(GpuSettings const& gpuSettings, SimulationData data)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsLauncher::shallowUpdateSelectedEntities(GpuSettings const& gpuSettings, SimulationData data, ShallowUpdateSelectionData updateData)
{
    bool reconnectionRequired = !updateData.considerClusters && (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.angleDelta != 0);

    //disconnect selection in case of reconnection
    if (reconnectionRequired) {
        int counter = 10;
        do {
            KERNEL_CALL_1_1(cudaPrepareForUpdate, data);

            setValueToDevice(_cudaUpdateResult, 0);
            KERNEL_CALL(cudaDisconnectSelection, data, _cudaUpdateResult);
            KERNEL_CALL(cudaProcessConnectionChanges, data);
            cudaDeviceSynchronize();
        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all affecting connections may be removed at first => repeat
    }

    if (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.velDeltaX != 0 || updateData.velDeltaY != 0) {
        KERNEL_CALL(cudaUpdatePosAndVelForSelection, updateData, data);
    }
    if (updateData.angleDelta != 0 || updateData.angularVelDelta != 0) {
        setValueToDevice(_cudaCenter, float2{0, 0});
        setValueToDevice(_cudaNumEntities, 0);
        KERNEL_CALL(cudaCalcAccumulatedCenter, updateData, data, _cudaCenter, _cudaNumEntities);
        cudaDeviceSynchronize();

        auto numEntities = copyToHost(_cudaNumEntities);
        if (numEntities != 0) {
            auto center = copyToHost(_cudaCenter);
            setValueToDevice(_cudaCenter, float2{center.x / numEntities, center.y / numEntities});
        }
        KERNEL_CALL(cudaUpdateAngleAndAngularVelForSelection, updateData, data, copyToHost(_cudaCenter));
    }

    //connect selection in case of reconnection
    if (reconnectionRequired) {
        cudaDeviceSynchronize();

        int counter = 10;
        do {
            KERNEL_CALL_1_1(cudaPrepareForUpdate, data);

            setValueToDevice(_cudaUpdateResult, 0);
            KERNEL_CALL(cudaUpdateMapForConnection, data);
            KERNEL_CALL(cudaConnectSelection, data, _cudaUpdateResult);
            KERNEL_CALL(cudaProcessConnectionChanges, data);

            KERNEL_CALL(cudaCleanupCellMap, data);
        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all necessary connections may be established at first => repeat

        updateSelection(gpuSettings, data);
    }
}

void _EditKernelsLauncher::rolloutSelection(GpuSettings const& gpuSettings, SimulationData data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        KERNEL_CALL(cudaRolloutSelectionStep, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}
