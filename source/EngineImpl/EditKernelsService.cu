#include "EditKernelsService.cuh"

#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/EditKernels.cuh>
#include <EngineKernels/SelectionKernels.cuh>
#include <EngineKernels/SimulationKernels.cuh>

#include "GarbageCollectorKernelsService.cuh"
#include "SelectionKernelsService.cuh"

void EditKernelsService::init()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _cudaRolloutResult);
    memoryManager.acquireMemory(1, _cudaSwitchResult);
    memoryManager.acquireMemory(1, _cudaUpdateResult);
    memoryManager.acquireMemory(1, _cudaRemoveResult);
    memoryManager.acquireMemory(1, _cudaInjectResult);
    memoryManager.acquireMemory(1, _cudaCenter);
    memoryManager.acquireMemory(1, _cudaVelocity);
    memoryManager.acquireMemory(1, _cudaNumEntities);
    memoryManager.acquireMemory(1, _cudaMinCellPosYAndIndex);
    memoryManager.acquireMemory(1, _genomePtr);
}

void EditKernelsService::shutdown()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_cudaRolloutResult);
    memoryManager.freeMemory(_cudaSwitchResult);
    memoryManager.freeMemory(_cudaUpdateResult);
    memoryManager.freeMemory(_cudaRemoveResult);
    memoryManager.freeMemory(_cudaInjectResult);
    memoryManager.freeMemory(_cudaCenter);
    memoryManager.freeMemory(_cudaVelocity);
    memoryManager.freeMemory(_cudaNumEntities);
    memoryManager.freeMemory(_cudaMinCellPosYAndIndex);
    memoryManager.freeMemory(_genomePtr);
}

void EditKernelsService::shallowUpdateSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, ShallowUpdateSelectionData const& updateData)
{
    bool reconnectionRequired = !updateData.considerClusters && (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.angleDelta != 0);

    //disconnect selection in case of reconnection
    if (reconnectionRequired) {
        int counter = 10;
        do {
            KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

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
        KERNEL_CALL(cudaCalcObjectWithMinimalPosY, data, _cudaMinCellPosYAndIndex);
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
            KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

            setValueToDevice(_cudaUpdateResult, 0);
            KERNEL_CALL(cudaPrepareMapForReconnection, data);
            KERNEL_CALL(cudaUpdateMapForReconnection, data);
            KERNEL_CALL(cudaScheduleConnectSelection, data, false, _cudaUpdateResult);
            KERNEL_CALL_1_1(cudaPrepareConnectionChanges, data);
            KERNEL_CALL(cudaProcessDeleteConnectionChanges, data);
            KERNEL_CALL(cudaProcessAddConnectionChanges, data);

            KERNEL_CALL(cudaCleanupMaps, data);
            cudaDeviceSynchronize();

        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all necessary connections may be established at first => repeat

        SelectionKernelsService::get().updateSelection(gpuSettings, data);
    }
}

void EditKernelsService::removeSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRemoveSelectedObjectConnections, data, includeClusters);

    KERNEL_CALL(cudaRemoveSelectedEntities, data, includeClusters);
    cudaDeviceSynchronize();

    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(gpuSettings, data);
}

void EditKernelsService::relaxSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRelaxSelectedEntities, data, includeClusters);
}

void EditKernelsService::uniformVelocities(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
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

void EditKernelsService::makeSticky(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaMakeSticky, data, includeClusters);
}

void EditKernelsService::removeStickiness(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRemoveStickiness, data, includeClusters);
}

void EditKernelsService::setBarrier(CudaSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters)
{
    KERNEL_CALL(cudaSetBarrier, data, value, includeClusters);
}

void EditKernelsService::reconnect(CudaSettings const& gpuSettings, SimulationData const& data)
{
    int counter = 10;
    do {
        KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

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
        KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

        setValueToDevice(_cudaUpdateResult, 0);
        KERNEL_CALL(cudaPrepareMapForReconnection, data);
        KERNEL_CALL(cudaUpdateMapForReconnection, data);
        KERNEL_CALL(cudaScheduleConnectSelection, data, false, _cudaUpdateResult);
        KERNEL_CALL_1_1(cudaPrepareConnectionChanges, data);
        KERNEL_CALL(cudaProcessDeleteConnectionChanges, data);
        KERNEL_CALL(cudaProcessAddConnectionChanges, data);

        KERNEL_CALL(cudaCleanupMaps, data);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  //due to locking not all necessary connections may be established at first => repeat

    SelectionKernelsService::get().updateSelection(gpuSettings, data);
}

void EditKernelsService::changeSimulationData(CudaSettings const& gpuSettings, SimulationData const& data, TOs const& changeTO)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    if (copyToHost(changeTO.numObjects) == 1) {
        KERNEL_CALL(cudaChangeObject, data, changeTO);
        cudaDeviceSynchronize();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());
    }
    if (copyToHost(changeTO.numEnergyParticles) == 1) {
        KERNEL_CALL(cudaChangeParticle, data, changeTO);
        cudaDeviceSynchronize();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());
    }
    cudaDeviceSynchronize();

    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(gpuSettings, data);
}

int EditKernelsService::injectGenomeToSelectedCreatures(CudaSettings const& gpuSettings, SimulationData const& data, TOs const& to)
{
    KERNEL_CALL_1_1(cudaCreateGenomeFromTO, data, to, _genomePtr);
    setValueToDevice(_cudaInjectResult, 0);
    KERNEL_CALL(cudaInjectGenomeToSelectedCreatures, data, _genomePtr, _cudaInjectResult);
    cudaDeviceSynchronize();
    return copyToHost(_cudaInjectResult);
}

void EditKernelsService::colorSelectedCells(CudaSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters)
{
    KERNEL_CALL(cudaColorSelectedObjects, data, color, includeClusters);
}

void EditKernelsService::setDetached(CudaSettings const& gpuSettings, SimulationData const& data, bool value)
{
    KERNEL_CALL(cudaSetDetached, data, value);
}

void EditKernelsService::applyForce(CudaSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData)
{
    KERNEL_CALL(cudaApplyForce, data, applyData);
}

void EditKernelsService::applyCataclysm(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaApplyCataclysm, data);
}

void EditKernelsService::getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult)
{
    KERNEL_CALL_1_1(cudaResetSelectionResult, selectionResult);
    setValueToDevice(_cudaMinCellPosYAndIndex, 0xffffffffffffffffull);
    KERNEL_CALL(cudaCalcObjectWithMinimalPosY, data, _cudaMinCellPosYAndIndex);
    cudaDeviceSynchronize();
    auto refCellIndex = static_cast<int>(copyToHost(_cudaMinCellPosYAndIndex) & 0xffffffff);
    KERNEL_CALL(cudaGetSelectionShallowData_step1, data);
    KERNEL_CALL(cudaGetSelectionShallowData_step2, data, refCellIndex, selectionResult);
    KERNEL_CALL_1_1(cudaFinalizeSelectionResult, selectionResult, data.objectMap);
}
