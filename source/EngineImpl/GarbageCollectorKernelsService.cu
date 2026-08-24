#include "GarbageCollectorKernelsService.cuh"

#include <EngineKernels/DebugKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>

void GarbageCollectorKernelsService::init()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

void GarbageCollectorKernelsService::shutdown()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void GarbageCollectorKernelsService::cleanupAfterTimestep(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaCleanupMaps), LaunchConfig{launchSettings.numBlocks, 8}, data);

    launchKernelOnDefaultStream(KERNEL(cudaPreparePointerArraysForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Energy*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.energies);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Object*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.objects);
    launchKernelOnDefaultStream(KERNEL(cudaSwapPointerArrays), LaunchConfig{1, 1}, data);

    launchKernelOnDefaultStream(KERNEL(cudaCheckIfCleanupIsNecessary), LaunchConfig{1, 1}, data, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        launchKernelOnDefaultStream(KERNEL(cudaPrepareHeapForCleanup), LaunchConfig{1, 1}, data);
        launchKernelOnDefaultStream(KERNEL(cudaCleanupParticles), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudaPrepareCleanupCreaturesAndGenomes), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects);
        launchKernelOnDefaultStream(KERNEL(cudaCleanupGenomesStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudacudaCleanupGenomesStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudaCleanupCreaturesStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudaCleanupCreaturesStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudaCleanupCellsStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudaCleanupCellsStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(
            KERNEL(cudaCleanupDependentCellData), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
        launchKernelOnDefaultStream(KERNEL(cudaSwapHeaps), LaunchConfig{1, 1}, data);
    }
}

void GarbageCollectorKernelsService::launchCleanupForPreviewInGraph(cudaStream_t stream, int numBlocks, SimulationData const& data)
{
    launchKernel(KERNEL(cudaPreparePointerArraysForCleanup), LaunchConfig{1, 1}, stream, data);
    ;
    launchKernel(KERNEL(cudaCleanupPointerArray<Energy*>), LaunchConfig{numBlocks, 8}, stream, data.entities.energies, data.tempEntities.energies);
    ;
    launchKernel(KERNEL(cudaCleanupPointerArray<Object*>), LaunchConfig{numBlocks, 8}, stream, data.entities.objects, data.tempEntities.objects);
    ;
    launchKernel(KERNEL(cudaSwapPointerArrays), LaunchConfig{1, 1}, stream, data);
    ;
    launchKernel(KERNEL(cudaCleanupMaps), LaunchConfig{numBlocks, 8}, stream, data);
    ;
}


void GarbageCollectorKernelsService::cleanupAfterDataManipulation(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaPreparePointerArraysForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Energy*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.energies);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Object*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.objects);
    launchKernelOnDefaultStream(KERNEL(cudaSwapPointerArrays), LaunchConfig{1, 1}, data);

    launchKernelOnDefaultStream(KERNEL(cudaPrepareHeapForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupParticles), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaPrepareCleanupCreaturesAndGenomes), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupGenomesStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudacudaCleanupGenomesStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCreaturesStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCreaturesStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCellsStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCellsStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupDependentCellData), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaSwapHeaps), LaunchConfig{1, 1}, data);
}

void GarbageCollectorKernelsService::copyArrays(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaPreparePointerArraysForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Energy*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.energies);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Object*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.objects);

    launchKernelOnDefaultStream(KERNEL(cudaPrepareHeapForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupParticles), LaunchConfig{launchSettings.numBlocks, 8}, data.tempEntities.energies, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaPrepareCleanupCreaturesAndGenomes), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupGenomesStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudacudaCleanupGenomesStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCreaturesStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCreaturesStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCellsStep1), LaunchConfig{launchSettings.numBlocks, 8}, data.tempEntities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(KERNEL(cudaCleanupCellsStep2), LaunchConfig{launchSettings.numBlocks, 8}, data.tempEntities.objects, data.tempEntities.heap);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupDependentCellData), LaunchConfig{launchSettings.numBlocks, 8}, data.tempEntities.objects, data.tempEntities.heap);
}

void GarbageCollectorKernelsService::swapArrays(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaSwapPointerArrays), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(KERNEL(cudaSwapHeaps), LaunchConfig{1, 1}, data);
}
