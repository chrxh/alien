#include "GarbageCollectorKernelsService.cuh"

#include <ranges>

#include <EngineKernels/DebugKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>

void GarbageCollectorKernelsService::init()
{
    getCudaBool();
}

void GarbageCollectorKernelsService::shutdown()
{
    for (auto& cudaBool : _cudaBools | std::views::values) {
        CudaMemoryManager::getInstance().freeMemory(cudaBool);
    }
    _cudaBools.clear();
}

bool* GarbageCollectorKernelsService::getCudaBool()
{
    auto& result = _cudaBools[getCurrentDevice()];
    if (!result) {
        CudaMemoryManager::getInstance().acquireMemory<bool>(1, result);
    }
    return result;
}

void GarbageCollectorKernelsService::cleanupAfterTimestep(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaCleanupGrids), LaunchConfig{launchSettings.numBlocks, 8}, data);

    launchKernelOnDefaultStream(KERNEL(cudaPreparePointerArraysForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Energy*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.energies);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Object*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.objects);
    launchKernelOnDefaultStream(KERNEL(cudaSwapPointerArrays), LaunchConfig{1, 1}, data);

    auto cudaBool = getCudaBool();
    launchKernelOnDefaultStream(KERNEL(cudaCheckIfCleanupIsNecessary), LaunchConfig{1, 1}, data, cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(cudaBool)) {
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
    launchKernel(KERNEL(cudaCleanupGrids), LaunchConfig{numBlocks, 8}, stream, data);
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

void GarbageCollectorKernelsService::compactPointerArrays(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaPreparePointerArraysForCleanup), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Energy*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.energies, data.tempEntities.energies);
    launchKernelOnDefaultStream(
        KERNEL(cudaCleanupPointerArray<Object*>), LaunchConfig{launchSettings.numBlocks, 8}, data.entities.objects, data.tempEntities.objects);
    launchKernelOnDefaultStream(KERNEL(cudaSwapPointerArrays), LaunchConfig{1, 1}, data);
}
