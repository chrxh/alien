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
    SimulationData const& data,
    SimulationResult const& result)
{
    KERNEL_CALL_1_1(cudaPrepareNextTimestep, data, result);
    if (flowFieldSettings.active) {
        KERNEL_CALL(cudaApplyFlowFieldSettings, data);
    }
    KERNEL_CALL(cudaNextTimestep_substep1, data);
    KERNEL_CALL(cudaNextTimestep_substep2, data);
    KERNEL_CALL(cudaNextTimestep_substep3, data);
    KERNEL_CALL(cudaNextTimestep_substep4, data);
    KERNEL_CALL(cudaNextTimestep_substep5, data);
    KERNEL_CALL(cudaNextTimestep_substep6, data, result);
    KERNEL_CALL(cudaNextTimestep_substep7, data);
    KERNEL_CALL(cudaNextTimestep_substep8, data, result);
    KERNEL_CALL(cudaNextTimestep_substep9, data);
    KERNEL_CALL(cudaNextTimestep_substep10, data);
    if (++_counter == 3) {
        KERNEL_CALL(cudaInitClusterData, data);
        KERNEL_CALL(cudaFindClusterIteration, data);  //3 iterations give a good approximation
        KERNEL_CALL(cudaFindClusterIteration, data);
        KERNEL_CALL(cudaFindClusterIteration, data);
        KERNEL_CALL(cudaFindClusterBoundaries, data);
        KERNEL_CALL(cudaAccumulateClusterPosAndVel, data);
        KERNEL_CALL(cudaAccumulateClusterAngularProp, data);
        KERNEL_CALL(cudaApplyClusterData, data);
        _counter = 0;
    }
    KERNEL_CALL_1_1(cudaNextTimestep_substep11, data);
    KERNEL_CALL(cudaNextTimestep_substep12, data);
    KERNEL_CALL(cudaNextTimestep_substep13, data);
    KERNEL_CALL(cudaNextTimestep_substep14, data);

    _garbageCollector->cleanupAfterTimestep(gpuSettings, data);
}
