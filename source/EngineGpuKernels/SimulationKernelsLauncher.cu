#include "SimulationKernelsLauncher.cuh"

#include "EngineInterface/SpaceCalculator.h"

#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"
#include "DebugKernels.cuh"
#include "SimulationStatistics.cuh"

_SimulationKernelsLauncher::_SimulationKernelsLauncher()
{
    _garbageCollector = std::make_shared<_GarbageCollectorKernelsLauncher>();
}

namespace 
{
    int calcOptimalThreadsForFluidKernel(SimulationParameters const& parameters)
    {
        auto scanRectLength = ceilf(parameters.smoothingLength.value * 2) * 2 + 1;
        return scanRectLength * scanRectLength;
    }
}

void _SimulationKernelsLauncher::calcTimestep(SettingsForSimulation const& settings, SimulationData const& data, SimulationStatistics const& statistics)
{
    auto const gpuSettings = settings.gpuSettings;
    KERNEL_CALL_1_1(cudaNextTimestep_prepare, data, statistics);

    // Not all kernels need to be executed in each time step for performance reasons
    bool considerForcesFromAngleDifferences = (data.timestep % 3 == 0);
    bool considerInnerFriction = (data.timestep % 3 == 0);
    bool considerRigidityUpdate = (data.timestep % 3 == 0);

    KERNEL_CALL(cudaNextTimestep_physics_init, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, 64, data);
    if (settings.simulationParameters.motionType.value == MotionType_Fluid) {
        auto threadBlockSize = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, threadBlockSize, data);
    } else {
        KERNEL_CALL(cudaNextTimestep_physics_calcCollisionForces, data);
    }
    if (settings.simulationParameters.numZones.value > 0) {
        KERNEL_CALL(cudaApplyFlowFieldSettings, data);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, 16, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, considerForcesFromAngleDifferences);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, 16, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, considerForcesFromAngleDifferences);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, 16, data);

    // Signal processing
    KERNEL_CALL(cudaNextTimestep_signal_calcFutureSignals, data);
    KERNEL_CALL(cudaNextTimestep_signal_updateSignals, data);
    KERNEL_CALL_MOD(cudaNextTimestep_signal_neuralNetworks, MAX_CHANNELS * MAX_CHANNELS, data, statistics);

    // Energy flow
    KERNEL_CALL_MOD(cudaNextTimestep_energyFlow, 32, data);

    // Cell type-specific functions
    KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep1, data);
    KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep2, data);
    KERNEL_CALL(cudaNextTimestep_cellType_oscillator, data, statistics);

    if (settings.simulationParameters.constructorCompletenessCheck) {
        KERNEL_CALL(cudaNextTimestep_cellType_constructor_completenessCheck, data, statistics);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_constructor, 4, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_injector, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_attacker, 4, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_transmitter, 4, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_muscle, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_sensor, 64, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_reconnector, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_detonator, data, statistics);

    if (considerInnerFriction) {
        KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, 16, data);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, 16, data);

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
    KERNEL_CALL_1_1(cudaNextTimestep_structuralOperations_substep1, data);
    KERNEL_CALL(cudaNextTimestep_structuralOperations_substep2, data);
    KERNEL_CALL(cudaNextTimestep_structuralOperations_substep3, data);
    KERNEL_CALL(cudaNextTimestep_structuralOperations_substep4, data);
    KERNEL_CALL(cudaNextTimestep_structuralOperations_substep5, data);

    _garbageCollector->cleanupAfterTimestep(settings.gpuSettings, data);
}

void _SimulationKernelsLauncher::prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& data)
{
    auto const gpuSettings = settings.gpuSettings;
    KERNEL_CALL(cudaResetDensity, data);
}

bool _SimulationKernelsLauncher::isRigidityUpdateEnabled(SettingsForSimulation const& settings) const
{
    for (int i = 0; i < settings.simulationParameters.numZones.value; ++i) {
        if (settings.simulationParameters.zone[i].values.rigidity != 0) {
            return true;
        }
    }
    return settings.simulationParameters.baseValues.rigidity != 0;
}
