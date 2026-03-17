#include "GarbageCollectorKernelsService.cuh"

#include <EngineGpuKernels/DebugKernels.cuh>

void GarbageCollectorKernelsService::init()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

void GarbageCollectorKernelsService::shutdown()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void GarbageCollectorKernelsService::cleanupAfterTimestep(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaCleanupMaps, data);

    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Energy*>, data.entities.energies, data.tempEntities.energies);
    KERNEL_CALL(cudaCleanupPointerArray<Object*>, data.entities.objects, data.tempEntities.objects);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaCheckIfCleanupIsNecessary, data, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
        KERNEL_CALL(cudaCleanupParticles, data.entities.energies, data.tempEntities.heap);
        KERNEL_CALL(cudaPrepareCleanupCreaturesAndGenomes, data.entities.objects);
        KERNEL_CALL(cudaCleanupGenomesStep1, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL(cudacudaCleanupGenomesStep2, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL(cudaCleanupCreaturesStep1, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL(cudaCleanupCreaturesStep2, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL(cudaCleanupCellsStep1, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL(cudaCleanupCellsStep2, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL(cudaCleanupDependentCellData, data.entities.objects, data.tempEntities.heap);
        KERNEL_CALL_1_1(cudaSwapHeaps, data);
    }
}

void GarbageCollectorKernelsService::cleanupAfterTimestepForPreview(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaCleanupMaps, data);

    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Energy*>, data.entities.energies, data.tempEntities.energies);
    KERNEL_CALL(cudaCleanupPointerArray<Object*>, data.entities.objects, data.tempEntities.objects);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);
}

void GarbageCollectorKernelsService::cleanupAfterDataManipulation(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Energy*>, data.entities.energies, data.tempEntities.energies);
    KERNEL_CALL(cudaCleanupPointerArray<Object*>, data.entities.objects, data.tempEntities.objects);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.entities.energies, data.tempEntities.heap);
    KERNEL_CALL(cudaPrepareCleanupCreaturesAndGenomes, data.entities.objects);
    KERNEL_CALL(cudaCleanupGenomesStep1, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudacudaCleanupGenomesStep2, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep1, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep2, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCellsStep1, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCellsStep2, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupDependentCellData, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL_1_1(cudaSwapHeaps, data);
}

void GarbageCollectorKernelsService::copyArrays(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Energy*>, data.entities.energies, data.tempEntities.energies);
    KERNEL_CALL(cudaCleanupPointerArray<Object*>, data.entities.objects, data.tempEntities.objects);

    KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.tempEntities.energies, data.tempEntities.heap);
    KERNEL_CALL(cudaPrepareCleanupCreaturesAndGenomes, data.entities.objects);
    KERNEL_CALL(cudaCleanupGenomesStep1, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudacudaCleanupGenomesStep2, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep1, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep2, data.entities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCellsStep1, data.tempEntities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupCellsStep2, data.tempEntities.objects, data.tempEntities.heap);
    KERNEL_CALL(cudaCleanupDependentCellData, data.tempEntities.objects, data.tempEntities.heap);
}

void GarbageCollectorKernelsService::swapArrays(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);
    KERNEL_CALL_1_1(cudaSwapHeaps, data);
}
