#include "GarbageCollectorKernelsLauncher.cuh"

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
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.entities.particlePointers, data.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.entities.cellPointers, data.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cudaCleanupPointerArray<Token*>, data.entities.tokenPointers, data.entitiesForCleanup.tokenPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaCheckIfCleanupIsNecessary, data, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, data);
        KERNEL_CALL(cudaCleanupParticles, data.entities.particlePointers, data.entitiesForCleanup.particles);
        KERNEL_CALL(cudaCleanupCellsStep1, data.entities.cellPointers, data.entitiesForCleanup.cells);
        KERNEL_CALL(cudaCleanupCellsStep2, data.entities.tokenPointers, data.entitiesForCleanup.cells);
        KERNEL_CALL(cudaCleanupTokens, data.entities.tokenPointers, data.entitiesForCleanup.tokens);
        KERNEL_CALL_1_1(cudaSwapArrays, data);
    }
}

void _GarbageCollectorKernelsLauncher::cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.entities.particlePointers, data.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.entities.cellPointers, data.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cudaCleanupPointerArray<Token*>, data.entities.tokenPointers, data.entitiesForCleanup.tokenPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);

    KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.entities.particlePointers, data.entitiesForCleanup.particles);
    KERNEL_CALL(cudaCleanupCellsStep1, data.entities.cellPointers, data.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupCellsStep2, data.entities.tokenPointers, data.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupTokens, data.entities.tokenPointers, data.entitiesForCleanup.tokens);
    KERNEL_CALL(cudaCleanupStringBytes, data.entities.cellPointers, data.entitiesForCleanup.stringBytes);
    KERNEL_CALL_1_1(cudaSwapArrays, data);
}

void _GarbageCollectorKernelsLauncher::copyArrays(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, data.entities.particlePointers, data.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, data.entities.cellPointers, data.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cudaCleanupPointerArray<Token*>, data.entities.tokenPointers, data.entitiesForCleanup.tokenPointers);

    KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, data);
    KERNEL_CALL(cudaCleanupParticles, data.entitiesForCleanup.particlePointers, data.entitiesForCleanup.particles);
    KERNEL_CALL(cudaCleanupCellsStep1, data.entitiesForCleanup.cellPointers, data.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupCellsStep2, data.entitiesForCleanup.tokenPointers, data.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupTokens, data.entitiesForCleanup.tokenPointers, data.entitiesForCleanup.tokens);
    KERNEL_CALL(cudaCleanupStringBytes, data.entitiesForCleanup.cellPointers, data.entitiesForCleanup.stringBytes);
}

void _GarbageCollectorKernelsLauncher::swapArrays(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL_1_1(cudaSwapPointerArrays, data);
    KERNEL_CALL_1_1(cudaSwapArrays, data);
}
