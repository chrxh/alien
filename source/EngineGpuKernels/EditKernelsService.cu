#include "DataAccessKernels.cuh"
#include "EditKernels.cuh"
#include "EditKernelsService.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "SimulationKernels.cuh"

_EditKernelsService::_EditKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _cudaRolloutResult);
    memoryManager.acquireMemory(1, _cudaSwitchResult);
    memoryManager.acquireMemory(1, _cudaUpdateResult);
    memoryManager.acquireMemory(1, _cudaRemoveResult);
    memoryManager.acquireMemory(1, _cudaCenter);
    memoryManager.acquireMemory(1, _cudaVelocity);
    memoryManager.acquireMemory(1, _cudaNumEntities);
    memoryManager.acquireMemory(1, _cudaMinCellPosYAndIndex);
    memoryManager.acquireMemory(1, _cudaMinCellPosYAndIndex);
    memoryManager.acquireMemory(1, _genomePtr);
    memoryManager.acquireMemory(1, _creaturePtr);
    memoryManager.acquireMemory(1, _result);
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsService>();
}

_EditKernelsService::~_EditKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_cudaRolloutResult);
    memoryManager.freeMemory(_cudaSwitchResult);
    memoryManager.freeMemory(_cudaUpdateResult);
    memoryManager.freeMemory(_cudaRemoveResult);
    memoryManager.freeMemory(_cudaCenter);
    memoryManager.freeMemory(_cudaVelocity);
    memoryManager.freeMemory(_cudaNumEntities);
    memoryManager.freeMemory(_cudaMinCellPosYAndIndex);
    memoryManager.freeMemory(_genomePtr);
    memoryManager.freeMemory(_creaturePtr);
    memoryManager.freeMemory(_result);
}

void _EditKernelsService::removeSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, false);
}

void _EditKernelsService::swapSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    KERNEL_CALL(cudaSwapSelection, switchData.pos, switchData.radius, data);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsService::switchSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData)
{
    setValueToDevice(_cudaSwitchResult, 0);

    KERNEL_CALL(cudaExistsSelection, switchData, data, _cudaSwitchResult);
    cudaDeviceSynchronize();

    if (0 == copyToHost(_cudaSwitchResult)) {
        KERNEL_CALL(cudaSetSelection, switchData.pos, switchData.radius, data);
        rolloutSelection(gpuSettings, data);
    }
}

void _EditKernelsService::setSelection(CudaSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData)
{
    KERNEL_CALL(cudaSetSelection, setData, data);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsService::updateSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaRemoveSelection, data, true);
    rolloutSelection(gpuSettings, data);
}

void _EditKernelsService::getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult)
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

void _EditKernelsService::shallowUpdateSelectedObjects(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    ShallowUpdateSelectionData const& updateData)
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

        updateSelection(gpuSettings, data);
    }
}

void _EditKernelsService::removeSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRemoveSelectedCellConnections, data, includeClusters);

    KERNEL_CALL(cudaRemoveSelectedEntities, data, includeClusters);
    cudaDeviceSynchronize();

    _garbageCollector->cleanupAfterDataManipulation(gpuSettings, data);
}

void _EditKernelsService::relaxSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRelaxSelectedEntities, data, includeClusters);
}

void _EditKernelsService::uniformVelocities(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
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

void _EditKernelsService::makeSticky(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaMakeSticky, data, includeClusters);
}

void _EditKernelsService::removeStickiness(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters)
{
    KERNEL_CALL(cudaRemoveStickiness, data, includeClusters);
}

void _EditKernelsService::setBarrier(CudaSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters)
{
    KERNEL_CALL(cudaSetBarrier, data, value, includeClusters);
}

void _EditKernelsService::reconnect(CudaSettings const& gpuSettings, SimulationData const& data)
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

    updateSelection(gpuSettings, data);
}

void _EditKernelsService::changeSimulationData(CudaSettings const& gpuSettings, SimulationData const& data, TO const& changeTO)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    if (copyToHost(changeTO.numCells) == 1) {
        KERNEL_CALL(cudaChangeCell, data, changeTO);
        cudaDeviceSynchronize();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());
    }
    if (copyToHost(changeTO.numParticles) == 1) {
        KERNEL_CALL(cudaChangeParticle, data, changeTO);
        cudaDeviceSynchronize();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());
    }
    cudaDeviceSynchronize();

    _garbageCollector->cleanupAfterDataManipulation(gpuSettings, data);
}

bool _EditKernelsService::changeCreature(CudaSettings const& gpuSettings, SimulationData const& data, TO const& to)
{
    setValueToDevice(_result, false);
    KERNEL_CALL_1_1(cudaAddGenomeAndCreature, data, to, _genomePtr, _creaturePtr);
    KERNEL_CALL(cudaChangeCellToCreature, data, _creaturePtr, _result);
    cudaDeviceSynchronize();

    return copyToHost(_result);
}

void _EditKernelsService::colorSelectedCells(CudaSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters)
{
    KERNEL_CALL(cudaColorSelectedCells, data, color, includeClusters);
}

void _EditKernelsService::setDetached(CudaSettings const& gpuSettings, SimulationData const& data, bool value)
{
    KERNEL_CALL(cudaSetDetached, data, value);
}

void _EditKernelsService::applyForce(CudaSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData)
{
    KERNEL_CALL(cudaApplyForce, data, applyData);
}

void _EditKernelsService::rolloutSelection(CudaSettings const& gpuSettings, SimulationData const& data)
{
    do {
        setValueToDevice(_cudaRolloutResult, 0);
        KERNEL_CALL(cudaRolloutSelectionStep, data, _cudaRolloutResult);
        cudaDeviceSynchronize();

    } while (1 == copyToHost(_cudaRolloutResult));
}

void _EditKernelsService::applyCataclysm(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaApplyCataclysm, data);
}
