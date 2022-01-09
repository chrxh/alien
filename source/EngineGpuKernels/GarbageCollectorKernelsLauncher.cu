#include "GarbageCollectorKernelsLauncher.cuh"

_GarbageCollectorKernelsLauncher::_GarbageCollectorKernelsLauncher()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

_GarbageCollectorKernelsLauncher::~_GarbageCollectorKernelsLauncher()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void _GarbageCollectorKernelsLauncher::cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL(cudaCleanupCellMap, simulationData);
    KERNEL_CALL(cudaCleanupParticleMap, simulationData);

    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, simulationData);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cudaCleanupPointerArray<Token*>, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokenPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, simulationData);
    cudaDeviceSynchronize();

    KERNEL_CALL_1_1(cudaCheckIfCleanupIsNecessary, simulationData, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, simulationData);
        KERNEL_CALL(cudaCleanupParticles, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particles);
        KERNEL_CALL(cudaCleanupCellsStep1, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cells);
        KERNEL_CALL(cudaCleanupCellsStep2, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.cells);
        KERNEL_CALL(cudaCleanupTokens, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokens);
        KERNEL_CALL_1_1(cudaSwapArrays, simulationData);
    }
}

void _GarbageCollectorKernelsLauncher::cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, simulationData);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cudaCleanupPointerArray<Token*>, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokenPointers);
    KERNEL_CALL_1_1(cudaSwapPointerArrays, simulationData);

    KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, simulationData);
    KERNEL_CALL(cudaCleanupParticles, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particles);
    KERNEL_CALL(cudaCleanupCellsStep1, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupCellsStep2, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupTokens, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokens);
    KERNEL_CALL_1_1(cudaSwapArrays, simulationData);
}

void _GarbageCollectorKernelsLauncher::copyArrays(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL_1_1(cudaPreparePointerArraysForCleanup, simulationData);
    KERNEL_CALL(cudaCleanupPointerArray<Particle*>, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cudaCleanupPointerArray<Cell*>, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cudaCleanupPointerArray<Token*>, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokenPointers);

    KERNEL_CALL_1_1(cudaPrepareArraysForCleanup, simulationData);
    KERNEL_CALL(cudaCleanupParticles, simulationData.entitiesForCleanup.particlePointers, simulationData.entitiesForCleanup.particles);
    KERNEL_CALL(cudaCleanupCellsStep1, simulationData.entitiesForCleanup.cellPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupCellsStep2, simulationData.entitiesForCleanup.tokenPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cudaCleanupTokens, simulationData.entitiesForCleanup.tokenPointers, simulationData.entitiesForCleanup.tokens);
}
