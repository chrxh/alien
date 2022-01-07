#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "GarbageCollectorKernels.cuh"

class GarbageCollectorKernelLauncher
{
public:
    GarbageCollectorKernelLauncher();
    ~GarbageCollectorKernelLauncher();

    void cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData);

private:
    bool* _cudaBool;
};

/**
 * Implementation
 */

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

    KERNEL_CALL_1_1(cleanupEntityArraysNecessary, simulationData, _cudaBool);
    cudaDeviceSynchronize();
    bool result;
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(&result, _cudaBool, sizeof(bool), cudaMemcpyDeviceToHost));

    if (result) {
        KERNEL_CALL_1_1(prepareArraysForCleanup, simulationData);
        KERNEL_CALL(cleanupParticles, simulationData.entities.particlePointers, simulationData.entitiesForCleanup.particles);
        KERNEL_CALL(cleanupCellsStep1, simulationData.entities.cellPointers, simulationData.entitiesForCleanup.cells);
        KERNEL_CALL(cleanupCellsStep2, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.cells);
        KERNEL_CALL(cleanupTokens, simulationData.entities.tokenPointers, simulationData.entitiesForCleanup.tokens);
        KERNEL_CALL_1_1(swapArrays, simulationData);
    }
}
