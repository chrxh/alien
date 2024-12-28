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
        auto scanRectLength = ceilf(parameters.motionData.fluidMotion.smoothingLength * 2) * 2 + 1;
        return scanRectLength * scanRectLength;
    }
}

void _SimulationKernelsLauncher::calcTimestep(Settings const& settings, SimulationData const& data, SimulationStatistics const& statistics)
{
    auto const gpuSettings = settings.gpuSettings;
    KERNEL_CALL_1_1(cudaNextTimestep_prepare, data, statistics);

    //not all kernels need to be executed in each time step for performance reasons
    bool considerForcesFromAngleDifferences = (data.timestep % 3 == 0);
    bool considerInnerFriction = (data.timestep % 3 == 0);
    bool considerRigidityUpdate = (data.timestep % 3 == 0);

    KERNEL_CALL(cudaNextTimestep_physics_init, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, 64, data);
    if (settings.simulationParameters.motionType == MotionType_Fluid) {
        auto threadBlockSize = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, threadBlockSize, data);
    } else {
        KERNEL_CALL(cudaNextTimestep_physics_calcCollisionForces, data);
    }
    if (settings.simulationParameters.numZones > 0) {
        KERNEL_CALL(cudaApplyFlowFieldSettings, data);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, 16, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, considerForcesFromAngleDifferences);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, 16, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, considerForcesFromAngleDifferences);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, 16, data);

    //cell functions
    KERNEL_CALL(cudaNextTimestep_cellFunction_prepare_substep1, data);
    KERNEL_CALL(cudaNextTimestep_cellFunction_prepare_substep2, data);
    KERNEL_CALL(cudaNextTimestep_cellFunction_nerve, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_neuron, data, statistics);
    if (settings.simulationParameters.cellFunctionConstructorCheckCompletenessForSelfReplication) {
        KERNEL_CALL(cudaNextTimestep_cellFunction_constructor_completenessCheck, data, statistics);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_cellFunction_constructor, 4, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_injector, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellFunction_attacker, 4, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellFunction_transmitter, 4, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_muscle, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellFunction_sensor, 64, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_reconnector, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_detonator, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_updateSignal, data, statistics);

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

void _SimulationKernelsLauncher::prepareForSimulationParametersChanges(Settings const& settings, SimulationData const& data)
{
    auto const gpuSettings = settings.gpuSettings;
    KERNEL_CALL(cudaResetDensity, data);
}

bool _SimulationKernelsLauncher::isRigidityUpdateEnabled(Settings const& settings) const
{
    for (int i = 0; i < settings.simulationParameters.numZones; ++i) {
        if (settings.simulationParameters.zone[i].values.rigidity != 0) {
            return true;
        }
    }
    return settings.simulationParameters.baseValues.rigidity != 0;
}
