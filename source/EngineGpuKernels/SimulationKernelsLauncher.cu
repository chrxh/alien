#include <cmath>
#include "SimulationKernelsLauncher.cuh"

#include "SimulationKernels.cuh"
#include "FlowFieldKernels.cuh"
#include "GarbageCollectorKernelsLauncher.cuh"
#include "DebugKernels.cuh"
#include "SimulationStatistics.cuh"
#include "EngineInterface/SpaceCalculator.h"

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
    KERNEL_CALL(cudaNextTimestep_physics_fillMaps, data);
    if (settings.simulationParameters.motionType == MotionType_Fluid) {
        auto threads = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
        cudaNextTimestep_physics_calcFluidForces<<<gpuSettings.numBlocks, threads>>>(data);
    } else {
        KERNEL_CALL(cudaNextTimestep_physics_calcCollisionForces, data);
    }
    if (settings.simulationParameters.numSpots > 0) {
        KERNEL_CALL(cudaApplyFlowFieldSettings, data);
    }
    KERNEL_CALL(cudaNextTimestep_physics_applyForces, data);
    KERNEL_CALL(cudaNextTimestep_physics_calcConnectionForces, data, considerForcesFromAngleDifferences);
    KERNEL_CALL(cudaNextTimestep_physics_verletPositionUpdate, data);
    KERNEL_CALL(cudaNextTimestep_physics_calcConnectionForces, data, considerForcesFromAngleDifferences);
    KERNEL_CALL(cudaNextTimestep_physics_verletVelocityUpdate, data);

    //cell functions
    KERNEL_CALL(cudaNextTimestep_cellFunction_prepare_substep1, data);
    KERNEL_CALL(cudaNextTimestep_cellFunction_prepare_substep2, data);
    KERNEL_CALL(cudaNextTimestep_cellFunction_nerve, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_neuron, data, statistics);
    if (settings.simulationParameters.cellFunctionConstructorCheckCompletenessForSelfReplication) {
        KERNEL_CALL(cudaNextTimestep_cellFunction_constructor_completenessCheck, data, statistics);
    }
    KERNEL_CALL(cudaNextTimestep_cellFunction_constructor_process, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_injector, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_attacker, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_transmitter, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_muscle, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_sensor, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_reconnector, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellFunction_detonator, data, statistics);

    if (considerInnerFriction) {
        KERNEL_CALL(cudaNextTimestep_physics_substep7_innerFriction, data);
    }
    KERNEL_CALL(cudaNextTimestep_physics_substep8, data);

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

bool _SimulationKernelsLauncher::updateSimulationParametersAfterTimestep(Settings& settings, SimulationData const& simulationData)
{
    auto changesMade = false;
    auto const& worldSizeX = settings.generalSettings.worldSizeX;
    auto const& worldSizeY = settings.generalSettings.worldSizeY;
    SpaceCalculator space({worldSizeX, worldSizeY});
    for (int i = 0; i < settings.simulationParameters.numParticleSources; ++i) {
        auto& source = settings.simulationParameters.particleSources[i];
        if (source.velX != 0) {
            source.posX += source.velX * settings.simulationParameters.timestepSize;
            changesMade = true;
        }
        if (source.velY != 0) {
            source.posY += source.velY * settings.simulationParameters.timestepSize;
            changesMade = true;
        }
        auto correctedPosition = space.getCorrectedPosition({source.posX, source.posY});
        source.posX = correctedPosition.x;
        source.posY = correctedPosition.y;
    }
    for (int i = 0; i < settings.simulationParameters.numSpots; ++i) {
        auto& spot = settings.simulationParameters.spots[i];
        if (spot.velX != 0) {
            spot.posX += spot.velX * settings.simulationParameters.timestepSize;
            changesMade = true;
        }
        if (spot.velY != 0) {
            spot.posY += spot.velY * settings.simulationParameters.timestepSize;
            changesMade = true;
        }
        auto correctedPosition = space.getCorrectedPosition({spot.posX, spot.posY});
        spot.posX = correctedPosition.x;
        spot.posY = correctedPosition.y;
    }

    auto externalEnergyPresent = false;
    for (int i = 0; i < MAX_COLORS; ++i) {
        externalEnergyPresent |= settings.simulationParameters.cellFunctionConstructorExternalEnergy[i] > 0;
    }
    if (externalEnergyPresent) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(
            &settings.simulationParameters.cellFunctionConstructorExternalEnergy,
            simulationData.externalEnergy,
            sizeof(ColorVector<float>),
            cudaMemcpyDeviceToHost));
        changesMade = true;
    }
    return changesMade;
}

void _SimulationKernelsLauncher::prepareForSimulationParametersChanges(Settings const& settings, SimulationData const& data)
{
    auto const gpuSettings = settings.gpuSettings;
    KERNEL_CALL(cudaResetDensity, data);
}

bool _SimulationKernelsLauncher::isRigidityUpdateEnabled(Settings const& settings) const
{
    for (int i = 0; i < settings.simulationParameters.numSpots; ++i) {
        if (settings.simulationParameters.spots[i].values.rigidity != 0) {
            return true;
        }
    }
    return settings.simulationParameters.baseValues.rigidity != 0;
}
