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

    //not all kernels need to be executed in each time step for performance reasons
    bool considerForcesFromAngleMismatch = (data.timestep % 3 == 0);
    bool considerInnerFriction = (data.timestep % 3 == 0);
    bool considerRigidityUpdate = (data.timestep % 3 == 0);

    KERNEL_CALL(cudaNextTimestep_substep1, data);
    KERNEL_CALL(cudaNextTimestep_substep2, data);
    KERNEL_CALL(cudaNextTimestep_substep3, data);
    KERNEL_CALL(cudaNextTimestep_connectionForces, data, considerForcesFromAngleMismatch);
    KERNEL_CALL(cudaNextTimestep_verletPositionUpdate, data);
    KERNEL_CALL(cudaNextTimestep_connectionForces, data, considerForcesFromAngleMismatch);
    KERNEL_CALL(cudaNextTimestep_verletVelocityUpdate, data);
    KERNEL_CALL(cudaNextTimestep_collectCellFunctionOperation, data);
    KERNEL_CALL(cudaNextTimestep_nerveFunction, data, result);
    KERNEL_CALL(cudaNextTimestep_neuronFunction, data, result);
    if (considerInnerFriction) {
        KERNEL_CALL(cudaNextTimestep_innerFriction, data);
    }
    KERNEL_CALL(cudaNextTimestep_frictionAndDecay, data);

    if (considerRigidityUpdate && isRigidityUpdateEnabled(settings)) {
        KERNEL_CALL(cudaInitClusterData, data);
        KERNEL_CALL(cudaFindClusterIteration, data);  //3 iterations should provide a good approximation
        KERNEL_CALL(cudaFindClusterIteration, data);
        KERNEL_CALL(cudaFindClusterIteration, data);
        KERNEL_CALL(cudaFindClusterBoundaries, data);
        KERNEL_CALL(cudaAccumulateClusterPosAndVel, data);
        KERNEL_CALL(cudaAccumulateClusterAngularProp, data);
        KERNEL_CALL(cudaApplyClusterData, data);
    }
    KERNEL_CALL_1_1(cudaNextTimestep_structuralOperations_step1, data);
    KERNEL_CALL(cudaNextTimestep_structuralOperations_step2, data);
    KERNEL_CALL(cudaNextTimestep_substep14, data);
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
