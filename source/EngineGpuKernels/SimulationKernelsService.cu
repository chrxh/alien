#include <EngineInterface/SpaceCalculator.h>

#include "DebugKernels.cuh"
#include "ForceFieldKernels.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "SimulationKernels.cuh"
#include "SimulationKernelsService.cuh"
#include "SimulationStatistics.cuh"

// =============================================================================
// CUDA 13 Performance Optimization Notes
// =============================================================================
// See docs/CUDA13_PERFORMANCE_OPTIMIZATION_EVALUATION.md for detailed analysis.
//
// High-priority optimization opportunities:
// 1. CUDA Graphs: Capture the kernel launch sequence to reduce CPU overhead
//    - The calcTimestep() method launches 35-40 kernels per timestep
//    - Graph capture could reduce launch overhead by 5-15%
//
// 2. Warp-Level Reductions: Use __shfl_down_sync before atomics
//    - Heavy atomic usage in physics kernels (force accumulation)
//    - Could improve physics kernel performance by 15-25%
//
// 3. Cooperative Groups: Enhanced synchronization in NeuronProcessor
//    - tile_partition and cg::reduce for efficient reductions
//    - Could improve neural network processing by 10-20%
//
// 4. Thread Block Clusters (CUDA 13): Cross-SM synchronization
//    - Could merge the 3 cudaFindClusterIteration calls into one kernel
//    - Potential 20-30% improvement for rigidity calculations
// =============================================================================

void SimulationKernelsService::init()
{
    // CUDA 13 Optimization: Initialize CUDA Graph structures here
    // Future: Create graph templates for different simulation modes
}

void SimulationKernelsService::shutdown()
{
    // CUDA 13 Optimization: Destroy CUDA Graph structures here
    // Future: Clean up graph instances and execution graphs
}

namespace
{
    int calcOptimalThreadsForFluidKernel(SimulationParameters const& parameters)
    {
        auto scanRectLength = ceilf(parameters.smoothingLength.value * 2) * 2 + 1;
        return scanRectLength * scanRectLength;
    }
}

void SimulationKernelsService::calcTimestep(SettingsForSimulation const& settings, SimulationData const& data, SimulationStatistics const& statistics)
{
    // CUDA 13 Optimization: Consider capturing this entire kernel sequence as a CUDA Graph
    // Benefits: Reduces kernel launch overhead (~5-15% performance gain)
    // Implementation: Use cudaStreamBeginCapture/cudaStreamEndCapture for graph capture
    // Note: Conditional branches (motion type, force fields, rigidity) require graph conditionals
    
    auto const gpuSettings = settings.cudaSettings;
    KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

    // Not all kernels need to be executed in each time step for performance reasons
    bool calcAngularForces = (data.timestep % 3 == 0);
    bool considerInnerFriction = (data.timestep % 3 == 0);
    bool considerRigidityUpdate = (data.timestep % 3 == 0);

    // === Physics Phase ===
    // CUDA 13 Optimization: Physics kernels use heavy atomics for force accumulation
    // Consider warp-level reductions (__shfl_down_sync) before atomicAdd operations
    KERNEL_CALL(cudaNextTimestep_physics_init, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, 64, data);
    if (settings.simulationParameters.motionType.value == MotionType_Fluid) {
        auto threadBlockSize = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, threadBlockSize, data);
    } else {
        KERNEL_CALL(cudaNextTimestep_physics_calcCollisionForces, data);
    }
    if (settings.simulationParameters.numLayers > 0) {
        KERNEL_CALL(cudaApplyForceFieldSettings, data);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, 16, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, calcAngularForces);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, 16, data);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, calcAngularForces);
    KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, 16, data);

    // === Signal Processing Phase ===
    // CUDA 13 Optimization: Neural network processing uses shared memory and atomicAdd_block
    // Consider using Cooperative Groups with tile_partition for efficient reductions
    // See NeuronProcessor.cuh for implementation details
    KERNEL_CALL(cudaNextTimestep_signal_calcFutureSignals, data);
    KERNEL_CALL(cudaNextTimestep_signal_updateSignals, data);
    KERNEL_CALL_MOD(cudaNextTimestep_signal_neuralNetworks, MAX_CHANNELS * MAX_CHANNELS, data, statistics);

    // Energy flow
    KERNEL_CALL_MOD(cudaNextTimestep_energyFlow, 32, data);

    // Cell type-specific functions
    KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep1, data);
    KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep2, data);
    KERNEL_CALL(cudaNextTimestep_cellType_generator, data, statistics);

    if (settings.simulationParameters.constructorCompletenessCheck.value) {
        KERNEL_CALL(cudaNextTimestep_cellType_constructor_completenessCheck, data, statistics);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_constructor, 4, data, statistics, false);
    KERNEL_CALL(cudaNextTimestep_cellType_injector, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_attacker, 4, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_depot, 4, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_muscle, data, statistics);
    KERNEL_CALL_MOD(cudaNextTimestep_cellType_sensor, 64, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_reconnector, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_detonator, data, statistics);
    KERNEL_CALL(cudaNextTimestep_cellType_digestor, data, statistics);

    if (considerInnerFriction) {
        KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, 16, data);
    }
    KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, 16, data);

    // === Rigidity Calculations ===
    // CUDA 13 Optimization: The 3 cudaFindClusterIteration calls could be merged using
    // Thread Block Clusters for cross-SM synchronization (potential 20-30% improvement)
    // Alternative: Use Cooperative Groups grid-wide sync with cudaLaunchCooperativeKernel
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

    GarbageCollectorKernelsService::get().cleanupAfterTimestep(settings.cudaSettings, data);
}

void SimulationKernelsService::calcTimestepForPreview(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics,
    bool detailSimulation)
{
    auto const gpuSettings = settings.cudaSettings;

    if (!detailSimulation) {

        KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

        // Not all kernels need to be executed in each time step for performance reasons
        bool considerForcesFromAngleDifferences = (data.timestep % 3 == 0);
        bool considerInnerFriction = (data.timestep % 3 == 0);

        KERNEL_CALL(cudaNextTimestep_physics_init, data);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, 64, data);
        {
            auto threadBlockSize = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
            KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, threadBlockSize, data);
        }
        KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, 16, data);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, considerForcesFromAngleDifferences);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, 16, data);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, 16, data, considerForcesFromAngleDifferences);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, 16, data);

        // Cell type-specific functions
        KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep1, data);
        KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep2, data);

        KERNEL_CALL_MOD(cudaNextTimestep_cellType_constructor, 4, data, statistics, true);

        if (considerInnerFriction) {
            KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, 16, data);
        }
        KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, 16, data);

        GarbageCollectorKernelsService::get().cleanupAfterTimestepForPreview(settings.cudaSettings, data);

    } else {
        KERNEL_CALL_1_1(cudaNextTimestep_prepare, data);

        // Not all kernels need to be executed in each time step for performance reasons
        bool considerForcesFromAngleDifferences = (data.timestep % 3 == 0);
        bool considerInnerFriction = (data.timestep % 3 == 0);

        KERNEL_CALL(cudaNextTimestep_physics_init, data);
        KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, 64, data);
        if (settings.simulationParameters.motionType.value == MotionType_Fluid) {
            auto threadBlockSize = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
            KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, threadBlockSize, data);
        } else {
            KERNEL_CALL(cudaNextTimestep_physics_calcCollisionForces, data);
        }
        if (settings.simulationParameters.numLayers > 0) {
            KERNEL_CALL(cudaApplyForceFieldSettings, data);
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
        KERNEL_CALL(cudaNextTimestep_cellType_generator, data, statistics);

        if (settings.simulationParameters.constructorCompletenessCheck.value) {
            KERNEL_CALL(cudaNextTimestep_cellType_constructor_completenessCheck, data, statistics);
        }
        KERNEL_CALL_MOD(cudaNextTimestep_cellType_constructor, 4, data, statistics, true);
        //KERNEL_CALL(cudaNextTimestep_cellType_injector, data, statistics);
        //KERNEL_CALL_MOD(cudaNextTimestep_cellType_attacker, 4, data, statistics);
        //KERNEL_CALL_MOD(cudaNextTimestep_cellType_depot, 4, data, statistics);
        KERNEL_CALL(cudaNextTimestep_cellType_muscle, data, statistics);
        //KERNEL_CALL_MOD(cudaNextTimestep_cellType_sensor, 64, data, statistics);
        //KERNEL_CALL(cudaNextTimestep_cellType_reconnector, data, statistics);
        //KERNEL_CALL(cudaNextTimestep_cellType_detonator, data, statistics);

        if (considerInnerFriction) {
            KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, 16, data);
        }
        KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, 16, data);

        //KERNEL_CALL_1_1(cudaNextTimestep_structuralOperations_substep1, data);
        //KERNEL_CALL(cudaNextTimestep_structuralOperations_substep2, data);
        //KERNEL_CALL(cudaNextTimestep_structuralOperations_substep3, data);
        //KERNEL_CALL(cudaNextTimestep_structuralOperations_substep4, data);
        //KERNEL_CALL(cudaNextTimestep_structuralOperations_substep5, data);

        GarbageCollectorKernelsService::get().cleanupAfterTimestep(settings.cudaSettings, data);
    }
}

void SimulationKernelsService::prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& data)
{
    auto const gpuSettings = settings.cudaSettings;
    KERNEL_CALL(cudaResetDensity, data);
}

bool SimulationKernelsService::isRigidityUpdateEnabled(SettingsForSimulation const& settings) const
{
    for (int i = 0; i < settings.simulationParameters.numLayers; ++i) {
        if (settings.simulationParameters.rigidity.layerValues[i].value != 0) {
            return true;
        }
    }
    return settings.simulationParameters.rigidity.baseValue != 0;
}
