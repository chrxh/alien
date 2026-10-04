#include "SimulationKernelsService.cuh"

#include <algorithm>
#include <ranges>

#include <Base/KernelTracer.h>

#include <Data/SpaceCalculator.h>

#include <EngineKernels/ForceFieldKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>
#include <EngineKernels/SimulationKernels.cuh>
#include <EngineKernels/SimulationStatistics.cuh>

#include "GarbageCollectorKernelsService.cuh"


void SimulationKernelsService::init()
{
    _graphCache.clear();
    _previewGraphCache.clear();
    _streams.clear();
}

void SimulationKernelsService::shutdown()
{
    for (cudaGraphExec_t& graphExec : _graphCache | std::views::values) {
        CHECK_FOR_DEVICE_ERRORS(cudaGraphExecDestroy(graphExec));
    }
    for (cudaGraphExec_t& graphExec : _previewGraphCache | std::views::values) {
        CHECK_FOR_DEVICE_ERRORS(cudaGraphExecDestroy(graphExec));
    }
    for (cudaStream_t& stream : _streams | std::views::values) {
        CHECK_FOR_DEVICE_ERRORS(cudaStreamDestroy(stream));
    }
    _graphCache.clear();
    _previewGraphCache.clear();
    _streams.clear();
}

cudaStream_t SimulationKernelsService::getStream(int key)
{
    auto findResult = _streams.find(key);
    if (findResult != _streams.end()) {
        return findResult->second;
    }
    cudaStream_t stream = nullptr;
    CHECK_FOR_DEVICE_ERRORS(cudaStreamCreate(&stream));
    _streams.emplace(key, stream);
    return stream;
}


namespace
{
    LaunchConfig calcFluidLaunchConfig(KernelLaunchSettings const& settings)
    {
        auto const warpsPerBlock = settings.fluidWarpsPerBlock;
        return LaunchConfig{std::max(1, settings.numBlocks / warpsPerBlock), warpsPerBlock * WARP_SIZE};
    }

    // Only energy particles near barriers are processed, so a smaller grid avoids the overhead of idle blocks
    LaunchConfig calcNearBarrierParticlesLaunchConfig(int numBlocks)
    {
        auto constexpr BlockSize = 128;
        return LaunchConfig{std::max(1, numBlocks / 16), BlockSize};
    }

    void launchGeneGraphKernels(cudaStream_t stream, int numBlocks, SimulationData const& data)
    {
        launchKernel(KERNEL(cudaNextTimestep_geneGraph_voidNodesUnreachableFromLastNode), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_geneGraph_removeCyclesNotThroughRoot), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_geneGraph_removeUnreachableGenesFromRoot), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_geneGraph_limitGenesWithSeparation), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_geneGraph_updateTransitiveNumCells), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data);
    }
}

CudaGraphConfig SimulationKernelsService::buildGraphConfig(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    uint64_t timestep,
    bool forceCellFunctionExecution) const
{
    CudaGraphConfig config;
    config.domainIndex = data.domain.index;
    config.isDecomposed = data.domain.numDomains > 1;
    config.timestepMod3 = toInt(timestep % 3);
    config.executeCellFunction = forceCellFunctionExecution ? true : timestep % TIMESTEPS_PER_CELL_FUNCTION == 0;
    config.hasLayers = settings.simulationParameters.numLayers > 0;
    config.rigidityEnabled = isRigidityUpdateEnabled(settings);
    config.numBlocks = settings.kernelLaunchSettings.numBlocks;
    return config;
}

void SimulationKernelsService::launchTimestepKernels(
    CudaGraphConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    auto stream = getStream(config.domainIndex);
    auto numBlocks = config.numBlocks;
    bool calcAngularForces = (config.timestepMod3 == 0);
    bool considerInnerFriction = (config.timestepMod3 == 0);
    bool considerRigidityUpdate = (config.timestepMod3 == 0);

    launchKernel(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, stream, data);

    launchKernel(KERNEL(cudaNextTimestep_physics_init), LaunchConfig{numBlocks, 8}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_physics_fillGrids), LaunchConfig{numBlocks, 64}, stream, data);

    launchKernel(KERNEL(cudaNextTimestep_physics_calcFluidForces), calcFluidLaunchConfig(settings.kernelLaunchSettings), stream, data);
    launchKernel(KERNEL(cudaNextTimestep_physics_calcFluidBoundaryForces), calcFluidLaunchConfig(settings.kernelLaunchSettings), stream, data);

    if (config.hasLayers) {
        launchKernel(KERNEL(cudaApplyForceFields), LaunchConfig{numBlocks, 8}, stream, data);
    }

    launchKernel(KERNEL(cudaNextTimestep_physics_applyForces), LaunchConfig{numBlocks, 16}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_physics_moveEnergyParticlesNearBarriers), calcNearBarrierParticlesLaunchConfig(numBlocks), stream, data);
    launchKernel(KERNEL(cudaNextTimestep_physics_calcConnectionForces), LaunchConfig{numBlocks, 16}, stream, data, calcAngularForces);
    launchKernel(KERNEL(cudaNextTimestep_physics_verletPositionUpdate), LaunchConfig{numBlocks, 16}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_physics_calcConnectionForces), LaunchConfig{numBlocks, 16}, stream, data, calcAngularForces);
    launchKernel(KERNEL(cudaNextTimestep_physics_verletVelocityUpdate), LaunchConfig{numBlocks, 16}, stream, data);

    // Energy flow
    launchKernel(KERNEL(cudaNextTimestep_energyFlow), LaunchConfig{numBlocks, 32}, stream, data);

    // Cell state transitions and front angle updates (run every timestep)
    launchKernel(KERNEL(cudaNextTimestep_cellState_substep1), LaunchConfig{numBlocks, 8}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_cellState_substep2), LaunchConfig{numBlocks, 8}, stream, data);


    // Signal processing and cell type-specific functions
    if (config.executeCellFunction) {
        launchKernel(KERNEL(cudaNextTimestep_signal_calcSignal), LaunchConfig{numBlocks, WARP_SIZE}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_signal_setSignal), LaunchConfig{numBlocks, 8}, stream, data);

        // Cell type-specific functions
        launchKernel(KERNEL(cudaNextTimestep_cellType_prepare_substep1), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_cellType_generator), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        // The constructor mutates the host genome before cloning; the mutation needs NEURAL_NET_INPUTS threads per block.
        launchKernel(KERNEL(cudaNextTimestep_constructor_mutate), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data, statistics, false);
        launchGeneGraphKernels(stream, numBlocks, data);
        launchKernel(KERNEL(cudaNextTimestep_constructor_construct), LaunchConfig{numBlocks, 1}, stream, data, statistics, false);
        launchKernel(KERNEL(cudaNextTimestep_constructor_countConstructorsNeedingEnergy), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_constructor_prepareExternalEnergyInflow), LaunchConfig{1, 1}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_constructor_provideExternalEnergy), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_cellType_injector), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_attacker), LaunchConfig{numBlocks, 4}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_depot), LaunchConfig{numBlocks, 4}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_muscle), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        if (config.isDecomposed) {
            launchKernel(KERNEL(cudaNextTimestep_cellType_sensor_decomposed), LaunchConfig{numBlocks, 64}, stream, data, statistics);
            launchKernel(KERNEL(cudaNextTimestep_domain_requestSensorContinuations), LaunchConfig{numBlocks, 64}, stream, data);
        } else {
            launchKernel(KERNEL(cudaNextTimestep_cellType_sensor), LaunchConfig{numBlocks, 64}, stream, data, statistics);
        }
        launchKernel(KERNEL(cudaNextTimestep_cellType_reconnector), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_detonator), LaunchConfig{numBlocks, 64}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_digestor), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_memory), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_communicator), LaunchConfig{numBlocks, 64}, stream, data, statistics);
        launchKernel(KERNEL(cudaNextTimestep_cellType_void), LaunchConfig{numBlocks, 8}, stream, data, statistics);
    }

    // Received in the last sync round and processed here, since they need the filled grids
    if (config.isDecomposed) {
        launchKernel(KERNEL(cudaNextTimestep_domain_applyShockWaves), LaunchConfig{numBlocks, 64}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_domain_scanSensorRequests), LaunchConfig{numBlocks, 64}, stream, data);
    }

    if (considerInnerFriction) {
        launchKernel(KERNEL(cudaNextTimestep_physics_applyInnerFriction), LaunchConfig{numBlocks, 16}, stream, data);
    }
    launchKernel(KERNEL(cudaNextTimestep_physics_applyFriction), LaunchConfig{numBlocks, 16}, stream, data, false);

    if (considerRigidityUpdate && config.rigidityEnabled) {
        launchKernel(KERNEL(cudaInitClusterData), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaFindClusterIteration), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaFindClusterIteration), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaFindClusterIteration), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaFindClusterBoundaries), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaAccumulateClusterPosAndVel), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaAccumulateClusterAngularProp), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaApplyClusterData), LaunchConfig{numBlocks, 8}, stream, data);
    }

    launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep1), LaunchConfig{numBlocks, 8}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep2), LaunchConfig{numBlocks, 8}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep3), LaunchConfig{numBlocks, 8}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep4), LaunchConfig{numBlocks, 8}, stream, data);
    launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep5), LaunchConfig{numBlocks, 8}, stream, data);

    launchKernel(KERNEL(cudaNextTimestep_incTimestep), LaunchConfig{1, 1}, stream, data);
}

cudaGraphExec_t SimulationKernelsService::captureTimestepGraph(
    CudaGraphConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    cudaGraph_t graph;

    auto stream = getStream(config.domainIndex);
    CHECK_FOR_DEVICE_ERRORS(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    launchTimestepKernels(config, settings, data, statistics);

    CHECK_FOR_DEVICE_ERRORS(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CHECK_FOR_DEVICE_ERRORS(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CHECK_FOR_DEVICE_ERRORS(cudaGraphDestroy(graph));

    _graphCache[config] = graphExec;
    return graphExec;
}

void SimulationKernelsService::calcTimestep(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics,
    uint64_t timestep,
    bool forceCellFunctionExecution)
{
    // Build configuration key for graph caching
    auto config = buildGraphConfig(settings, data, timestep, forceCellFunctionExecution);
    auto stream = getStream(config.domainIndex);

    // In debug mode, bypass CUDA Graphs to get precise kernel crash information
    if (GlobalSettings::get().isDebugMode()) {
        KernelTracer::get().setTimestep(timestep);
        launchTimestepKernels(config, settings, data, statistics);
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream));
    } else {
        // Check if we have a cached graph for this configuration
        cudaGraphExec_t graphExec;
        auto it = _graphCache.find(config);
        if (it == _graphCache.end()) {
            // Capture a new graph for this configuration
            graphExec = captureTimestepGraph(config, settings, data, statistics);
        } else {
            graphExec = it->second;
        }

        // Execute the cached graph
        CHECK_FOR_DEVICE_ERRORS(cudaGraphLaunch(graphExec, stream));

        // Wait for the graph to complete before garbage collection
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream));
    }

    // Garbage collection cannot be part of the graph due to dynamic behavior
    GarbageCollectorKernelsService::get().cleanupAfterTimestep(settings.kernelLaunchSettings, data);
}

CudaGraphPreviewConfig SimulationKernelsService::buildPreviewGraphConfig(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    uint64_t timestep,
    bool forceCellFunctionExecution,
    bool detailSimulation) const
{
    CudaGraphPreviewConfig config;
    config.timestepMod3 = toInt(timestep % 3);
    config.executeCellFunctions = forceCellFunctionExecution ? true : timestep % TIMESTEPS_PER_CELL_FUNCTION == 0;
    config.detailSimulation = detailSimulation;
    config.numBlocks = settings.kernelLaunchSettings.numBlocks;
    return config;
}

void SimulationKernelsService::launchPreviewKernels(
    CudaGraphPreviewConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    auto stream = getStream(PreviewStreamKey);
    auto numBlocks = config.numBlocks;
    bool considerForcesFromAngleDifferences = (config.timestepMod3 == 0);
    bool considerInnerFriction = (config.timestepMod3 == 0);

    if (!config.detailSimulation) {
        launchKernel(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, stream, data);

        launchKernel(KERNEL(cudaNextTimestep_physics_init), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_fillGrids), LaunchConfig{numBlocks, 64}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcFluidForces), calcFluidLaunchConfig(settings.kernelLaunchSettings), stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcFluidBoundaryForces), calcFluidLaunchConfig(settings.kernelLaunchSettings), stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_applyForces), LaunchConfig{numBlocks, 16}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_moveEnergyParticlesNearBarriers), calcNearBarrierParticlesLaunchConfig(numBlocks), stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcConnectionForces), LaunchConfig{numBlocks, 16}, stream, data, considerForcesFromAngleDifferences);
        launchKernel(KERNEL(cudaNextTimestep_physics_verletPositionUpdate), LaunchConfig{numBlocks, 16}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcConnectionForces), LaunchConfig{numBlocks, 16}, stream, data, considerForcesFromAngleDifferences);
        launchKernel(KERNEL(cudaNextTimestep_physics_verletVelocityUpdate), LaunchConfig{numBlocks, 16}, stream, data);

        // Cell state transitions and front angle updates (run every timestep)
        launchKernel(KERNEL(cudaNextTimestep_cellState_substep1), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_cellState_substep2), LaunchConfig{numBlocks, 8}, stream, data);

        if (config.executeCellFunctions) {
            // Cell type-specific functions
            launchKernel(KERNEL(cudaNextTimestep_cellType_prepare_substep1), LaunchConfig{numBlocks, 8}, stream, data);

            launchKernel(KERNEL(cudaNextTimestep_constructor_mutate), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data, statistics, true);
            launchGeneGraphKernels(stream, numBlocks, data);
            launchKernel(KERNEL(cudaNextTimestep_constructor_construct), LaunchConfig{numBlocks, 1}, stream, data, statistics, true);
        }

        if (considerInnerFriction) {
            launchKernel(KERNEL(cudaNextTimestep_physics_applyInnerFriction), LaunchConfig{numBlocks, 16}, stream, data);
        }
        launchKernel(KERNEL(cudaNextTimestep_physics_applyFriction), LaunchConfig{numBlocks, 16}, stream, data, true);

        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep1), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep2), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep3), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep4), LaunchConfig{numBlocks, 8}, stream, data);

        GarbageCollectorKernelsService::get().launchCleanupForPreviewInGraph(stream, numBlocks, data);

        launchKernel(KERNEL(cudaNextTimestep_incTimestep), LaunchConfig{1, 1}, stream, data);
    } else {
        launchKernel(KERNEL(cudaNextTimestep_prepare), LaunchConfig{1, 1}, stream, data);

        launchKernel(KERNEL(cudaNextTimestep_physics_init), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_fillGrids), LaunchConfig{numBlocks, 64}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcFluidForces), calcFluidLaunchConfig(settings.kernelLaunchSettings), stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcFluidBoundaryForces), calcFluidLaunchConfig(settings.kernelLaunchSettings), stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_applyForces), LaunchConfig{numBlocks, 16}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_moveEnergyParticlesNearBarriers), calcNearBarrierParticlesLaunchConfig(numBlocks), stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcConnectionForces), LaunchConfig{numBlocks, 16}, stream, data, considerForcesFromAngleDifferences);
        launchKernel(KERNEL(cudaNextTimestep_physics_verletPositionUpdate), LaunchConfig{numBlocks, 16}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_physics_calcConnectionForces), LaunchConfig{numBlocks, 16}, stream, data, considerForcesFromAngleDifferences);
        launchKernel(KERNEL(cudaNextTimestep_physics_verletVelocityUpdate), LaunchConfig{numBlocks, 16}, stream, data);

        // Energy flow
        launchKernel(KERNEL(cudaNextTimestep_energyFlow), LaunchConfig{numBlocks, 32}, stream, data);

        // Cell state transitions and front angle updates (run every timestep)
        launchKernel(KERNEL(cudaNextTimestep_cellState_substep1), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_cellState_substep2), LaunchConfig{numBlocks, 8}, stream, data);

        if (config.executeCellFunctions) {
            // Signal processing
            launchKernel(KERNEL(cudaNextTimestep_signal_calcSignal), LaunchConfig{numBlocks, WARP_SIZE}, stream, data, statistics);
            launchKernel(KERNEL(cudaNextTimestep_signal_setSignal), LaunchConfig{numBlocks, 8}, stream, data);

            // Cell type-specific functions
            launchKernel(KERNEL(cudaNextTimestep_cellType_prepare_substep1), LaunchConfig{numBlocks, 8}, stream, data);
            launchKernel(KERNEL(cudaNextTimestep_cellType_generator), LaunchConfig{numBlocks, 8}, stream, data, statistics);

            launchKernel(KERNEL(cudaNextTimestep_constructor_mutate), LaunchConfig{numBlocks, NEURAL_NET_INPUTS}, stream, data, statistics, true);
            launchGeneGraphKernels(stream, numBlocks, data);
            launchKernel(KERNEL(cudaNextTimestep_constructor_construct), LaunchConfig{numBlocks, 1}, stream, data, statistics, true);
            launchKernel(KERNEL(cudaNextTimestep_cellType_muscle), LaunchConfig{numBlocks, 8}, stream, data, statistics);
            launchKernel(KERNEL(cudaNextTimestep_cellType_void), LaunchConfig{numBlocks, 8}, stream, data, statistics);
        }

        if (considerInnerFriction) {
            launchKernel(KERNEL(cudaNextTimestep_physics_applyInnerFriction), LaunchConfig{numBlocks, 16}, stream, data);
        }
        launchKernel(KERNEL(cudaNextTimestep_physics_applyFriction), LaunchConfig{numBlocks, 16}, stream, data, true);

        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep1), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep2), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep3), LaunchConfig{numBlocks, 8}, stream, data);
        launchKernel(KERNEL(cudaNextTimestep_structuralOperations_substep4), LaunchConfig{numBlocks, 8}, stream, data);

        GarbageCollectorKernelsService::get().launchCleanupForPreviewInGraph(stream, numBlocks, data);

        launchKernel(KERNEL(cudaNextTimestep_incTimestep), LaunchConfig{1, 1}, stream, data);
    }
}

cudaGraphExec_t SimulationKernelsService::capturePreviewGraph(
    CudaGraphPreviewConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    cudaGraph_t graph;

    auto stream = getStream(PreviewStreamKey);
    CHECK_FOR_DEVICE_ERRORS(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    launchPreviewKernels(config, settings, data, statistics);

    CHECK_FOR_DEVICE_ERRORS(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t graphExec;
    CHECK_FOR_DEVICE_ERRORS(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CHECK_FOR_DEVICE_ERRORS(cudaGraphDestroy(graph));

    _previewGraphCache[config] = graphExec;
    return graphExec;
}

void SimulationKernelsService::calcTimestepForPreview(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics,
    uint64_t timestep,
    bool forceCellFunctionExecution,
    bool detailSimulation)
{
    // Build configuration key for graph caching
    auto config = buildPreviewGraphConfig(settings, data, timestep, forceCellFunctionExecution, detailSimulation);
    auto stream = getStream(PreviewStreamKey);

    // In debug mode, bypass CUDA Graphs to get precise kernel crash information
    if (GlobalSettings::get().isDebugMode()) {
        KernelTracer::get().setTimestep(timestep);
        launchPreviewKernels(config, settings, data, statistics);
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream));
    } else {
        // Check if we have a cached graph for this configuration
        cudaGraphExec_t graphExec;
        auto it = _previewGraphCache.find(config);
        if (it == _previewGraphCache.end()) {
            // Capture a new graph for this configuration
            graphExec = capturePreviewGraph(config, settings, data, statistics);
        } else {
            graphExec = it->second;
        }

        // Execute the cached graph
        CHECK_FOR_DEVICE_ERRORS(cudaGraphLaunch(graphExec, stream));

        // Wait for the graph to complete before garbage collection
        CHECK_FOR_DEVICE_ERRORS(cudaStreamSynchronize(stream));
    }
}

void SimulationKernelsService::prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& data)
{
    // Invalidate graph cache when simulation parameters change
    // The cache will be rebuilt lazily on next calcTimestep call
    for (auto& pair : _graphCache) {
        CHECK_FOR_DEVICE_ERRORS(cudaGraphExecDestroy(pair.second));
    }
    _graphCache.clear();

    // Also invalidate preview graph cache
    for (auto& pair : _previewGraphCache) {
        CHECK_FOR_DEVICE_ERRORS(cudaGraphExecDestroy(pair.second));
    }
    _previewGraphCache.clear();

    auto const launchSettings = settings.kernelLaunchSettings;
    launchKernelOnDefaultStream(KERNEL(cudaResetDensity), LaunchConfig{launchSettings.numBlocks, 8}, data);
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
