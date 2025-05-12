﻿#include "GarbageCollectorKernelsLauncher.cuh"

_GarbageCollectorKernelsLauncher::_GarbageCollectorKernelsLauncher()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

_GarbageCollectorKernelsLauncher::~_GarbageCollectorKernelsLauncher()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void _GarbageCollectorKernelsLauncher::cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& data)
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
        KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, data);
        KERNEL_CALL(cudaCleanupParticles, data.objects.particlePointers, data.tempObjects.rawMemory);
        KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cellPointers, data.tempObjects.rawMemory);
        KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cellPointers, data.tempObjects.rawMemory);
        KERNEL_CALL(cudaCleanupRawMemory, data.objects.cellPointers, data.tempObjects.rawMemory);
        KERNEL_CALL_1_1(cudaSwapRawMemory, data);
    }
}

void _GarbageCollectorKernelsLauncher::cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particlePointers, data.tempObjects.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cellPointers, data.tempObjects.cellPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.objects.particlePointers, data.tempObjects.rawMemory);
    KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cellPointers, data.tempObjects.rawMemory);
    KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cellPointers, data.tempObjects.rawMemory);
    KERNEL_CALL(cudaCleanupRawMemory, data.objects.cellPointers, data.tempObjects.rawMemory);
    KERNEL_CALL_1_1(cudaSwapRawMemory, data);
}

void _GarbageCollectorKernelsLauncher::copyArrays(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.objects.particlePointers, data.tempObjects.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.objects.cellPointers, data.tempObjects.cellPointers);

    KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.tempObjects.particlePointers, data.tempObjects.rawMemory);
    KERNEL_CALL(cudaCleanupCellsStep1, data.objects.cellPointers, data.tempObjects.rawMemory);
    KERNEL_CALL(cudaCleanupCellsStep2, data.objects.cellPointers, data.tempObjects.rawMemory);
    KERNEL_CALL(cudaCleanupRawMemory, data.tempObjects.cellPointers, data.tempObjects.rawMemory);
}

void _GarbageCollectorKernelsLauncher::swapArrays(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);
    KERNEL_CALL_1_1(cudaSwapRawMemory, data);
}
