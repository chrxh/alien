#include "SimulationKernelsLauncher.cuh"

#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"

_SimulationKernelsLauncher::_SimulationKernelsLauncher()
{
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsLauncher>();
}

void _SimulationKernelsLauncher::calcTimestep(
    GpuSettings const& gpuSettings,
    FlowFieldSettings const& flowFieldSettings,
    SimulationData const& simulationData,
    SimulationResult const& result)
{
    KERNEL_CALL_1_1(prepareForNextTimestep, simulationData, result);
    if (flowFieldSettings.active) {
        KERNEL_CALL(cudaApplyFlowFieldSettings, simulationData);
    }
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
    if (++_counter == 3) {
        KERNEL_CALL(cudaInitClusterData, simulationData);
        KERNEL_CALL(cudaFindClusterIteration, simulationData);
        KERNEL_CALL(cudaFindClusterIteration, simulationData);
        KERNEL_CALL(cudaFindClusterIteration, simulationData);
        KERNEL_CALL(cudaFindClusterBoundaries, simulationData);
        KERNEL_CALL(cudaAccumulateClusterPosAndVel, simulationData);
        KERNEL_CALL(cudaAccumulateClusterAngularProp, simulationData);
        KERNEL_CALL(cudaApplyClusterData, simulationData);
        _counter = 0;
    }
    KERNEL_CALL(processingStep11, simulationData);
    KERNEL_CALL(processingStep12, simulationData);
    KERNEL_CALL(processingStep13, simulationData);
    KERNEL_CALL(processingStep14, simulationData);

    _garbageCollector->cleanupAfterTimestep(gpuSettings, simulationData);
}
