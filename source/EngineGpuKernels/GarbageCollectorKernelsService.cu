#include "GarbageCollectorKernelsService.cuh"

_GarbageCollectorKernelsService::_GarbageCollectorKernelsService()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

_GarbageCollectorKernelsService::~_GarbageCollectorKernelsService()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void _GarbageCollectorKernelsService::cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaCleanupCellMap, data);
    KERNEL_CALL(cudaCleanupParticleMap, data);

    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particlePointers, data.tempObjects.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cellPointers, data.tempObjects.cellPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaCheckIfCleanupIsNecessary, data, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
        KERNEL_CALL(cudaCleanupParticles, data.objects.particlePointers, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cellPointers, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cellPointers, data.tempObjects.heap);
        KERNEL_CALL(cudaCleanupDependentCellData, data.objects.cellPointers, data.tempObjects.heap);
        KERNEL_CALL_1_1(cudaSwapHeaps, data);
    }
}

void _GarbageCollectorKernelsService::cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particlePointers, data.tempObjects.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cellPointers, data.tempObjects.cellPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.objects.particlePointers, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cellPointers, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cellPointers, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupDependentCellData, data.objects.cellPointers, data.tempObjects.heap);
    KERNEL_CALL_1_1(cudaSwapHeaps, data);
}

void _GarbageCollectorKernelsService::copyArrays(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particlePointers, data.tempObjects.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cellPointers, data.tempObjects.cellPointers);

    KERNEL_CALL_1_1(cudaPrepareHeapForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.tempObjects.particlePointers, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep1, data.tempObjects.cellPointers, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupCellsStep2, data.tempObjects.cellPointers, data.tempObjects.heap);
    KERNEL_CALL(cudaCleanupDependentCellData, data.tempObjects.cellPointers, data.tempObjects.heap);
}

void _GarbageCollectorKernelsService::swapArrays(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);
    KERNEL_CALL_1_1(cudaSwapHeaps, data);
}
