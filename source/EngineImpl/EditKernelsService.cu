#include "EditKernelsService.cuh"

#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/EditKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>
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

void EditKernelsService::shallowUpdateSelectedObjects(
    KernelLaunchSettings const& gpuSettings,
    SimulationData const& data,
    ShallowUpdateSelectionData const& updateData)
{
    bool reconnectionRequired = !updateData.considerClusters && (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.angleDelta != 0);

    // Disconnect selection in case of reconnection
    if (reconnectionRequired) {
        int counter = 10;
        do {
            launchKernelOnDefaultStream(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, data);

            setValueToDevice(_cudaUpdateResult, 0);
            launchKernelOnDefaultStream(KERNEL(cudaScheduleDisconnectSelectionFromRemainings), LaunchConfig{gpuSettings.numBlocks, 8}, data, _cudaUpdateResult);
            launchKernelOnDefaultStream(KERNEL(cudaPrepareConnectionChanges), LaunchConfig{1, 1}, data);
            launchKernelOnDefaultStream(KERNEL(cudaProcessDeleteConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);
            launchKernelOnDefaultStream(KERNEL(cudaProcessAddConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);
            cudaDeviceSynchronize();
        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  // Due to locking not all affecting connections may be removed at first => repeat
    }

    if (updateData.posDeltaX != 0 || updateData.posDeltaY != 0 || updateData.velX != 0 || updateData.velY != 0) {
        launchKernelOnDefaultStream(KERNEL(cudaIncrementPosAndVelForSelection), LaunchConfig{gpuSettings.numBlocks, 8}, updateData, data);
    }
    if (updateData.angleDelta != 0 || updateData.angularVel != 0) {
        setValueToDevice(_cudaCenter, float2{0, 0});
        setValueToDevice(_cudaNumEntities, 0);

        setValueToDevice(_cudaMinCellPosYAndIndex, 0xffffffff00000000ull);
        launchKernelOnDefaultStream(KERNEL(cudaCalcObjectWithMinimalPosY), LaunchConfig{gpuSettings.numBlocks, 8}, data, _cudaMinCellPosYAndIndex);
        cudaDeviceSynchronize();
        auto refCellIndex = static_cast<int>(copyToHost(_cudaMinCellPosYAndIndex) & 0xffffffff);

        launchKernelOnDefaultStream(
            KERNEL(cudaCalcAccumulatedCenterAndVel),
            LaunchConfig{gpuSettings.numBlocks, 8},
            data,
            refCellIndex,
            _cudaCenter,
            nullptr,
            _cudaNumEntities,
            updateData.considerClusters);
        cudaDeviceSynchronize();

        auto numEntities = copyToHost(_cudaNumEntities);
        if (numEntities != 0) {
            auto center = copyToHost(_cudaCenter);
            setValueToDevice(_cudaCenter, float2{center.x / numEntities, center.y / numEntities});
        }
        launchKernelOnDefaultStream(
            KERNEL(cudaUpdateAngleAndAngularVelForSelection), LaunchConfig{gpuSettings.numBlocks, 8}, updateData, data, copyToHost(_cudaCenter));
    }

    // Connect selection in case of reconnection
    if (reconnectionRequired) {
        cudaDeviceSynchronize();

        int counter = 10;
        do {
            launchKernelOnDefaultStream(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, data);

            setValueToDevice(_cudaUpdateResult, 0);
            launchKernelOnDefaultStream(KERNEL(cudaPrepareMapForReconnection), LaunchConfig{gpuSettings.numBlocks, 8}, data);
            launchKernelOnDefaultStream(KERNEL(cudaUpdateMapForReconnection), LaunchConfig{gpuSettings.numBlocks, 8}, data);
            launchKernelOnDefaultStream(KERNEL(cudaScheduleConnectSelection), LaunchConfig{gpuSettings.numBlocks, 8}, data, false, _cudaUpdateResult);
            launchKernelOnDefaultStream(KERNEL(cudaPrepareConnectionChanges), LaunchConfig{1, 1}, data);
            launchKernelOnDefaultStream(KERNEL(cudaProcessDeleteConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);
            launchKernelOnDefaultStream(KERNEL(cudaProcessAddConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);

            launchKernelOnDefaultStream(KERNEL(cudaCleanupMaps), LaunchConfig{gpuSettings.numBlocks, 8}, data);
            cudaDeviceSynchronize();

        } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  // Due to locking not all necessary connections may be established at first => repeat

        SelectionKernelsService::get().updateSelection(gpuSettings, data);
    }
}

void EditKernelsService::removeSelectedObjects(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelectedObjectConnections), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters);

    launchKernelOnDefaultStream(KERNEL(cudaRemoveSelectedEntities), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters);
    cudaDeviceSynchronize();

    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(gpuSettings, data);
}

void EditKernelsService::relaxSelectedObjects(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    launchKernelOnDefaultStream(KERNEL(cudaRelaxSelectedEntities), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters);
}

void EditKernelsService::uniformVelocities(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    setValueToDevice(_cudaVelocity, float2{0, 0});
    setValueToDevice(_cudaNumEntities, 0);
    launchKernelOnDefaultStream(
        KERNEL(cudaCalcAccumulatedCenterAndVel), LaunchConfig{gpuSettings.numBlocks, 8}, data, -1, nullptr, _cudaVelocity, _cudaNumEntities, includeClusters);
    cudaDeviceSynchronize();

    auto numEntities = copyToHost(_cudaNumEntities);
    if (numEntities != 0) {
        auto velocity = copyToHost(_cudaVelocity) / numEntities;
        launchKernelOnDefaultStream(KERNEL(cudaSetVelocityForSelection), LaunchConfig{gpuSettings.numBlocks, 8}, data, velocity, includeClusters);
    }
}

void EditKernelsService::makeSticky(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    launchKernelOnDefaultStream(KERNEL(cudaMakeSticky), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters);
}

void EditKernelsService::removeStickiness(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    launchKernelOnDefaultStream(KERNEL(cudaRemoveStickiness), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters);
}

void EditKernelsService::setBarrier(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters)
{
    launchKernelOnDefaultStream(KERNEL(cudaSetBarrier), LaunchConfig{gpuSettings.numBlocks, 8}, data, value, includeClusters);
}

void EditKernelsService::reconnect(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    int counter = 10;
    do {
        launchKernelOnDefaultStream(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, data);

        setValueToDevice(_cudaUpdateResult, 0);
        launchKernelOnDefaultStream(KERNEL(cudaScheduleDisconnectSelectionFromRemainings), LaunchConfig{gpuSettings.numBlocks, 8}, data, _cudaUpdateResult);
        launchKernelOnDefaultStream(KERNEL(cudaPrepareConnectionChanges), LaunchConfig{1, 1}, data);
        launchKernelOnDefaultStream(KERNEL(cudaProcessDeleteConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);
        launchKernelOnDefaultStream(KERNEL(cudaProcessAddConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);
        cudaDeviceSynchronize();
    } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  // Due to locking not all affecting connections may be removed at first => repeat

    cudaDeviceSynchronize();

    counter = 10;
    do {
        launchKernelOnDefaultStream(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, data);

        setValueToDevice(_cudaUpdateResult, 0);
        launchKernelOnDefaultStream(KERNEL(cudaPrepareMapForReconnection), LaunchConfig{gpuSettings.numBlocks, 8}, data);
        launchKernelOnDefaultStream(KERNEL(cudaUpdateMapForReconnection), LaunchConfig{gpuSettings.numBlocks, 8}, data);
        launchKernelOnDefaultStream(KERNEL(cudaScheduleConnectSelection), LaunchConfig{gpuSettings.numBlocks, 8}, data, false, _cudaUpdateResult);
        launchKernelOnDefaultStream(KERNEL(cudaPrepareConnectionChanges), LaunchConfig{1, 1}, data);
        launchKernelOnDefaultStream(KERNEL(cudaProcessDeleteConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);
        launchKernelOnDefaultStream(KERNEL(cudaProcessAddConnectionChanges), LaunchConfig{gpuSettings.numBlocks, 8}, data);

        launchKernelOnDefaultStream(KERNEL(cudaCleanupMaps), LaunchConfig{gpuSettings.numBlocks, 8}, data);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaUpdateResult) && --counter > 0);  // Due to locking not all necessary connections may be established at first => repeat

    SelectionKernelsService::get().updateSelection(gpuSettings, data);
}

void EditKernelsService::changeSimulationData(KernelLaunchSettings const& gpuSettings, SimulationData const& data, TOs const& changeTO)
{
    launchKernelOnDefaultStream(KERNEL(cudaSaveNumEntries), LaunchConfig{1, 1}, data);

    cudaDeviceSynchronize();
    CHECK_FOR_DEVICE_ERRORS(cudaGetLastError());

    if (copyToHost(changeTO.numObjects) == 1) {
        launchKernelOnDefaultStream(KERNEL(cudaChangeObject), LaunchConfig{gpuSettings.numBlocks, 8}, data, changeTO);
        cudaDeviceSynchronize();
        CHECK_FOR_DEVICE_ERRORS(cudaGetLastError());
    }
    if (copyToHost(changeTO.numEnergyParticles) == 1) {
        launchKernelOnDefaultStream(KERNEL(cudaChangeParticle), LaunchConfig{gpuSettings.numBlocks, 8}, data, changeTO);
        cudaDeviceSynchronize();
        CHECK_FOR_DEVICE_ERRORS(cudaGetLastError());
    }
    cudaDeviceSynchronize();

    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(gpuSettings, data);
}

int EditKernelsService::injectGenomeToSelectedCreatures(KernelLaunchSettings const& gpuSettings, SimulationData const& data, TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaAdaptNumberGenerator), LaunchConfig{gpuSettings.numBlocks, 8}, data.primaryNumberGen, to);
    launchKernelOnDefaultStream(KERNEL(cudaCreateGenomeFromTO), LaunchConfig{1, 1}, data, to, _genomePtr);
    setValueToDevice(_cudaInjectResult, 0);
    launchKernelOnDefaultStream(KERNEL(cudaInjectGenomeToSelectedCreatures), LaunchConfig{gpuSettings.numBlocks, 8}, data, _genomePtr, _cudaInjectResult);
    cudaDeviceSynchronize();
    return copyToHost(_cudaInjectResult);
}

void EditKernelsService::colorSelectedCells(KernelLaunchSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters)
{
    launchKernelOnDefaultStream(KERNEL(cudaColorSelectedObjects), LaunchConfig{gpuSettings.numBlocks, 8}, data, color, includeClusters);
}

void EditKernelsService::setDetached(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool value)
{
    launchKernelOnDefaultStream(KERNEL(cudaSetDetached), LaunchConfig{gpuSettings.numBlocks, 8}, data, value);
}

void EditKernelsService::applyForce(KernelLaunchSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData)
{
    launchKernelOnDefaultStream(KERNEL(cudaApplyForce), LaunchConfig{gpuSettings.numBlocks, 8}, data, applyData);
}

void EditKernelsService::applyCataclysm(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaApplyCataclysm), LaunchConfig{gpuSettings.numBlocks, 8}, data);
}

void EditKernelsService::getSelectionShallowData(KernelLaunchSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult)
{
    launchKernelOnDefaultStream(KERNEL(cudaResetSelectionResult), LaunchConfig{1, 1}, selectionResult);
    setValueToDevice(_cudaMinCellPosYAndIndex, 0xffffffffffffffffull);
    launchKernelOnDefaultStream(KERNEL(cudaCalcObjectWithMinimalPosY), LaunchConfig{gpuSettings.numBlocks, 8}, data, _cudaMinCellPosYAndIndex);
    cudaDeviceSynchronize();
    auto refCellIndex = static_cast<int>(copyToHost(_cudaMinCellPosYAndIndex) & 0xffffffff);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectionShallowData_step1), LaunchConfig{gpuSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectionShallowData_step2), LaunchConfig{gpuSettings.numBlocks, 8}, data, refCellIndex, selectionResult);
    launchKernelOnDefaultStream(KERNEL(cudaFinalizeSelectionResult), LaunchConfig{1, 1}, selectionResult, data.objectMap);
}
