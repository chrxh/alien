#include "GarbageCollectorKernelsService.cuh"

#include "DebugKernels.cuh"

_GarbageCollectorKernelsService::_GarbageCollectorKernelsService()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

_GarbageCollectorKernelsService::~_GarbageCollectorKernelsService()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void _GarbageCollectorKernelsService::cleanupAfterTimestep(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaCleanupMaps, data);

    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particles, data.tempObjects.particles);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cells, data.tempObjects.cells);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaCheckIfCleanupIsNecessary, data, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
        KERNEL_CALL(cudaCleanupParticles, data.objects.particles, data.tempObjects.heap);
        KERNEL_CALL(cudaPrepareCleanupCreaturesAndGenomes, data.objects.cells);
        KERNEL_CALL(cudaCleanupGenomesStep1, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL(cudacudaCleanupGenomesStep2, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupCreaturesStep1, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupCreaturesStep2, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupDependentCellData, data.objects.cells, data.tempObjects.heap);
        KERNEL_CALL_1_1(cudaSwapHeaps, data);
    }
}

void _GarbageCollectorKernelsService::cleanupAfterTimestepForPreview(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaCleanupMaps, data);
}

void _GarbageCollectorKernelsService::cleanupAfterDataManipulation(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particles, data.tempObjects.particles);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cells, data.tempObjects.cells);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.objects.particles, data.tempObjects.heap);
    KERNEL_CALL(cudaPrepareCleanupCreaturesAndGenomes, data.objects.cells);
    KERNEL_CALL(cudaCleanupGenomesStep1, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudacudaCleanupGenomesStep2, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep1, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep2, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupDependentCellData, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL_1_1(cudaSwapHeaps, data);

    printf("-AAAAAA1---\n");
    KERNEL_CALL(DEBUG_checkCellsAndParticles, data, nullptr, 0);
    printf("-AAAAAA2---\n");
}

void _GarbageCollectorKernelsService::copyArrays(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particles, data.tempObjects.particles);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cells, data.tempObjects.cells);

    KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.tempObjects.particles, data.tempObjects.heap);
    KERNEL_CALL(cudaPrepareCleanupCreaturesAndGenomes, data.objects.cells);
    KERNEL_CALL(cudaCleanupGenomesStep1, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudacudaCleanupGenomesStep2, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep1, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCreaturesStep2, data.objects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep1, data.tempObjects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep2, data.tempObjects.cells, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupDependentCellData, data.tempObjects.cells, data.tempObjects.heap);
}

void _GarbageCollectorKernelsService::swapArrays(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);
    KERNEL_CALL_1_1(cudaSwapHeaps, data);
}
