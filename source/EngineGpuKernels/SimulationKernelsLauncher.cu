#include "SimulationKernelsLauncher.cuh"

#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"

_SimulationKernelsLauncher::_SimulationKernelsLauncher()
{
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsLauncher>();
}

void _SimulationKernelsLauncher::calcTimestep(Settings const& settings, SimulationData const& data, SimulationResult const& result)
{
    auto const gpuSettings = settings.gpuSettings;
    KERNEL_CALL_1_1(cudaPrepareNextTimestep, data, result);
    if (settings.flowFieldSettings.active) {
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

    if (isRigidityUpdateEnabled(settings)) {
        if (++_counter == 3) {  //execute rigidity update only every 3rd time step for performance reasons
            KERNEL_CALL(cudaInitClusterData, data);
            KERNEL_CALL(cudaFindClusterIteration, data);  //3 iterations should provide a good approximation
            KERNEL_CALL(cudaFindClusterIteration, data);
            KERNEL_CALL(cudaFindClusterIteration, data);
            KERNEL_CALL(cudaFindClusterBoundaries, data);
            KERNEL_CALL(cudaAccumulateClusterPosAndVel, data);
            KERNEL_CALL(cudaAccumulateClusterAngularProp, data);
            KERNEL_CALL(cudaApplyClusterData, data);
            _counter = 0;
        }
    }
    KERNEL_CALL_1_1(cudaNextTimestep_substep11, data);
    KERNEL_CALL(cudaNextTimestep_substep12, data);
    KERNEL_CALL(cudaNextTimestep_substep13, data);
    KERNEL_CALL(cudaNextTimestep_substep14, data);

    _garbageCollector->cleanupAfterTimestep(settings.gpuSettings, data);
}

bool _SimulationKernelsLauncher::isRigidityUpdateEnabled(Settings const& settings) const
{
    for(int i = 0; i < settings.simulationParametersSpots.numSpots; ++i) {
        if (settings.simulationParametersSpots.spots[i].values.rigidity != 0) {
            return true;
        }
    }
    return settings.simulationParameters.spotValues.rigidity != 0;
}
