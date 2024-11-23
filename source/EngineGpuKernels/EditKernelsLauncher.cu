﻿#include "EditKernelsLauncher.cuh"

#include "DataAccessKernels.cuh"
#include "EditKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"

_EditKernelsLauncher::_EditKernelsLauncher()
{
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaRolloutResult);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaSwitchResult);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaUpdateResult);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaRemoveResult);
    CudaMemoryManager::getInstance().acquireMemory<float2>(1, _cudaCenter);
    CudaMemoryManager::getInstance().acquireMemory<float2>(1, _cudaVelocity);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaNumEntities);
    CudaMemoryManager::getInstance().acquireMemory<unsigned long long int>(1, _cudaMinCellPosYAndIndex);
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsLauncher>();
}

_EditKernelsLauncher::~_EditKernelsLauncher()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaRolloutResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaSwitchResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaUpdateResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaRemoveResult);
    CudaMemoryManager::getInstance().freeMemory(_cudaCenter);
    CudaMemoryManager::getInstance().freeMemory(_cudaVelocity);
    CudaMemoryManager::getInstance().freeMemory(_cudaNumEntities);
    CudaMemoryManager::getInstance().freeMemory(_cudaMinCellPosYAndIndex);
}

void _EditKernelsLauncher::removeSelection(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, false);
}

void _EditKernelsLauncher::swapSelection(GpuSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    KERNEL_CALL(cudaSwapSelection, switchData.pos, switchData.radius, data);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsLauncher::switchSelection(GpuSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    KERNEL_CALL(cudaExistsSelection, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        KERNEL_CALL(cudaSetSelection, switchData.pos, switchData.radius, data);
        rolloutSelection(gpuSettings, data);
    }
}

void _EditKernelsLauncher::setSelection(GpuSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData)
{
    KERNEL_CALL(cudaSetSelection, setData, data);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsLauncher::updateSelection(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsLauncher::getSelectionShallowData(GpuSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult)
{
    KERNEL_CALL_1_1(cudaResetSelectionResult, selectionResult);
    setValueToDevice(_cudaMinCellPosYAndIndex, 0xffffffffffffffffull);
    KERNEL_CALL(cudaCalcCellWithMinimalPosY, data, _cudaMinCellPosYAndIndex);
    cudaDeviceSynchronize();
    auto refCellIndex = static_cast<int>(copyToHost(_cudaMinCellPosYAndIndex) & 0xffffffff);
    KERNEL_CALL(cudaGetSelectionShallowData, data, refCellIndex, selectionResult);
    KERNEL_CALL_1_1(cudaFinalizeSelectionResult, selectionResult, data.cellMap);
}

void _EditKernelsLauncher::shallowUpdateSelectedObjects(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    ShallowUpdateSelectionData const& updateData)
{
    bool reconnectionRequired = !updateData.considerClusters && (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.angleDelta != 0);

    //disconnect selection in case of reconnection
    if (reconnectionRequired) {
        int counter = 10;
        do {
            KERNEL_CALL_1_1(cudaPrepareForUpdate, data);

            setValueToDevice(_cudaUpdateResult, 0);
            KERNEL_CALL(cudaScheduleDisconnectSelectionFromRemainings, data, _cudaUpdateResult);
            KERNEL_CALL_1_1(cudaPrepareConnectionChanges, data);
            KERNEL_CALL(cudaProcessDeleteConnectionChanges, data);
            KERNEL_CALL(cudaProcessAddConnectionChanges, data);
            cudaDeviceSynchronize();
        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all affecting connections may be removed at first => repeat
    }

    if (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.velX != 0 || updateData.velY != 0) {
        KERNEL_CALL(cudaIncrementPosAndVelForSelection, updateData, data);
    }
    if (updateData.angleDelta != 0 || updateData.angularVel != 0) {
        setValueToDevice(_cudaCenter, float2{0, 0});
        setValueToDevice(_cudaNumEntities, 0);

        setValueToDevice(_cudaMinCellPosYAndIndex, 0xffffffff00000000ull);
        KERNEL_CALL(cudaCalcCellWithMinimalPosY, data, _cudaMinCellPosYAndIndex);
        cudaDeviceSynchronize();
        auto refCellIndex = static_cast<int>(copyToHost(_cudaMinCellPosYAndIndex) & 0xffffffff);

        KERNEL_CALL(cudaCalcAccumulatedCenterAndVel, data, refCellIndex, _cudaCenter, nullptr, _cudaNumEntities, updateData.considerClusters);
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
            KERNEL_CALL(cudaPrepareMapForReconnection, data);
            KERNEL_CALL(cudaUpdateMapForReconnection, data);
            KERNEL_CALL(cudaScheduleConnectSelection, data, false, _cudaUpdateResult);
            KERNEL_CALL_1_1(cudaPrepareConnectionChanges, data);
            KERNEL_CALL(cudaProcessDeleteConnectionChanges, data);
            KERNEL_CALL(cudaProcessAddConnectionChanges, data);

            KERNEL_CALL(cudaCleanupCellMap, data);
            cudaDeviceSynchronize();

        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all necessary connections may be established at first => repeat

        updateSelection(gpuSettings, data);
    }
}

void _EditKernelsLauncher::removeSelectedObjects(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRemoveSelectedCellConnections, data, includeClusters);

    KERNEL_CALL(cudaRemoveSelectedEntities, data, includeClusters);
    cudaDeviceSynchronize();
    
    _garbageCollector->cleanupAfterDataManipulation(gpuSettings, data);
}

void _EditKernelsLauncher::relaxSelectedObjects(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRelaxSelectedEntities, data, includeClusters);
}

void _EditKernelsLauncher::uniformVelocities(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    setValueToDevice(_cudaVelocity, float2{0, 0});
    setValueToDevice(_cudaNumEntities, 0);
    KERNEL_CALL(cudaCalcAccumulatedCenterAndVel, data, -1, nullptr, _cudaVelocity, _cudaNumEntities, includeClusters);
    cudaDeviceSynchronize();

    auto numEntities = copyToHost(_cudaNumEntities);
    if (numEntities != 0) {
        auto velocity = copyToHost(_cudaVelocity) / numEntities;
        KERNEL_CALL(cudaSetVelocityForSelection, data, velocity, includeClusters);
    }
}

void _EditKernelsLauncher::makeSticky(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaMakeSticky, data, includeClusters);
}

void _EditKernelsLauncher::removeStickiness(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRemoveStickiness, data, includeClusters);
}

void _EditKernelsLauncher::setBarrier(GpuSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters)
{
    KERNEL_CALL(cudaSetBarrier, data, value, includeClusters);
}

void _EditKernelsLauncher::reconnect(GpuSettings const& gpuSettings, SimulationData const& data)
{
    int counter = 10;
    do {
        KERNEL_CALL_1_1(cudaPrepareForUpdate, data);

        setValueToDevice(_cudaUpdateResult, 0);
        KERNEL_CALL(cudaScheduleDisconnectSelectionFromRemainings, data, _cudaUpdateResult);
        KERNEL_CALL_1_1(cudaPrepareConnectionChanges, data);
        KERNEL_CALL(cudaProcessDeleteConnectionChanges, data);
        KERNEL_CALL(cudaProcessAddConnectionChanges, data);
        cudaDeviceSynchronize();
    } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all affecting connections may be removed at first => repeat

        cudaDeviceSynchronize();

    counter = 10;
    do {
        KERNEL_CALL_1_1(cudaPrepareForUpdate, data);

        setValueToDevice(_cudaUpdateResult, 0);
        KERNEL_CALL(cudaPrepareMapForReconnection, data);
        KERNEL_CALL(cudaUpdateMapForReconnection, data);
        KERNEL_CALL(cudaScheduleConnectSelection, data, false, _cudaUpdateResult);
        KERNEL_CALL_1_1(cudaPrepareConnectionChanges, data);
        KERNEL_CALL(cudaProcessDeleteConnectionChanges, data);
        KERNEL_CALL(cudaProcessAddConnectionChanges, data);

        KERNEL_CALL(cudaCleanupCellMap, data);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all necessary connections may be established at first => repeat

    updateSelection(gpuSettings, data);
}

void _EditKernelsLauncher::changeSimulationData(GpuSettings const& gpuSettings, SimulationData const& data, DataTO const& changeDataTO)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    if (copyToHost(changeDataTO.numCells) == 1) {
        KERNEL_CALL(cudaChangeCell, data, changeDataTO);
        cudaDeviceSynchronize();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    }
    if (copyToHost(changeDataTO.numParticles) == 1) {
        KERNEL_CALL(cudaChangeParticle, data, changeDataTO);
        cudaDeviceSynchronize();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    }
    cudaDeviceSynchronize();

    _garbageCollector->cleanupAfterDataManipulation(gpuSettings, data);
}

void _EditKernelsLauncher::colorSelectedCells(GpuSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters)
{
    KERNEL_CALL(cudaColorSelectedCells, data, color, includeClusters);
}

void _EditKernelsLauncher::setDetached(GpuSettings const& gpuSettings, SimulationData const& data, bool value)
{
    KERNEL_CALL(cudaSetDetached, data, value);
}

void _EditKernelsLauncher::applyForce(GpuSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData)
{
    KERNEL_CALL(cudaApplyForce, data, applyData);
}

void _EditKernelsLauncher::rolloutSelection(GpuSettings const& gpuSettings, SimulationData const& data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        KERNEL_CALL(cudaRolloutSelectionStep, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}

void _EditKernelsLauncher::applyCataclysm(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaApplyCataclysm, data);
}
