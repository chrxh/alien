#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.cuh"
#include "Macros.cuh"
#include "SimulationKernels.cuh"

class SimulationKernelLauncher
{
public:
    static void calcTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData, SimulationResult const& result);
};

/**
 * Implementation
 */
void SimulationKernelLauncher::calcTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData, SimulationResult const& result)
{
    KERNEL_CALL_1_1(prepareForNextTimestep, simulationData, result);
    KERNEL_CALL_1_1(cudaApplyFlowFieldSettings, simulationData);

    KERNEL_CALL(processingStep1, simulationData);
    KERNEL_CALL(processingStep2, simulationData);
    KERNEL_CALL(processingStep3, simulationData);
    KERNEL_CALL(processingStep4, simulationData);
    KERNEL_CALL(processingStep5, simulationData);
    KERNEL_CALL(processingStep6, simulationData, result);
    KERNEL_CALL(processingStep7, simulationData);
    KERNEL_CALL(processingStep8, simulationData, result);
    KERNEL_CALL(processingStep9, simulationData);
    KERNEL_CALL(processingStep10, simulationData);
    KERNEL_CALL(processingStep11, simulationData);
    KERNEL_CALL(processingStep12, simulationData);
    KERNEL_CALL(processingStep13, simulationData);
    KERNEL_CALL_1_1(cleanupAfterSimulationKernel, simulationData);

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
}
