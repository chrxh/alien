#include "GarbageCollectorKernelLauncher.cuh"

GarbageCollectorKernelLauncher::GarbageCollectorKernelLauncher()
{
    CudaMemoryManager::getInstance().acquireMemory<bool>(1, _cudaBool);
}

GarbageCollectorKernelLauncher::~GarbageCollectorKernelLauncher()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaBool);
}

void GarbageCollectorKernelLauncher::cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL(cleanupCellMap, simulationData);
    KERNEL_CALL(cleanupParticleMap, simulationData);

    KERNEL_CALL_1_1(preparePointerArraysForCleanup, simulationData);
    KERNEL_CALL(cleanupPointerArray<Particle*>, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cleanupPointerArray<Cell*>, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cleanupPointerArray<Token*>, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokenPointers);
    KERNEL_CALL_1_1(swapPointerArrays, simulationData);
    cudaDeviceSynchronize();

    KERNEL_CALL_1_1(checkIfCleanupIsNecessary, simulationData, _cudaBool);
    cudaDeviceSynchronize();
    if (copyToHost(_cudaBool)) {
        KERNEL_CALL_1_1(prepareArraysForCleanup, simulationData);
        KERNEL_CALL(cleanupParticles, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particles);
        KERNEL_CALL(cleanupCellsStep1, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cells);
        KERNEL_CALL(cleanupCellsStep2, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.cells);
        KERNEL_CALL(cleanupTokens, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokens);
        KERNEL_CALL_1_1(swapArrays, simulationData);
    }
}

void GarbageCollectorKernelLauncher::cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL_1_1(preparePointerArraysForCleanup, simulationData);
    KERNEL_CALL(cleanupPointerArray<Particle*>, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cleanupPointerArray<Cell*>, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cleanupPointerArray<Token*>, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokenPointers);
    KERNEL_CALL_1_1(swapPointerArrays, simulationData);

    KERNEL_CALL_1_1(prepareArraysForCleanup, simulationData);
    KERNEL_CALL(cleanupParticles, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particles);
    KERNEL_CALL(cleanupCellsStep1, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cleanupCellsStep2, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cleanupTokens, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokens);
    KERNEL_CALL_1_1(swapArrays, simulationData);
}

void GarbageCollectorKernelLauncher::copyArrays(GpuSettings const& gpuSettings, SimulationData const& simulationData)
{
    KERNEL_CALL_1_1(preparePointerArraysForCleanup, simulationData);
    KERNEL_CALL(cleanupPointerArray<Particle*>, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particlePointers);
    KERNEL_CALL(cleanupPointerArray<Cell*>, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cellPointers);
    KERNEL_CALL(cleanupPointerArray<Token*>, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokenPointers);

    KERNEL_CALL_1_1(prepareArraysForCleanup, simulationData);
    KERNEL_CALL(cleanupParticles, simulationData.entitiesForCleanup.particlePointers, simulationData.entitiesForCleanup.particles);
    KERNEL_CALL(cleanupCellsStep1, simulationData.entitiesForCleanup.cellPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cleanupCellsStep2, simulationData.entitiesForCleanup.tokenPointers, simulationData.entitiesForCleanup.cells);
    KERNEL_CALL(cleanupTokens, simulationData.entitiesForCleanup.tokenPointers, simulationData.entitiesForCleanup.tokens);
}
