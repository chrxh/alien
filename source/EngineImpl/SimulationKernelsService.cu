#include "SimulationKernelsService.cuh"

#include <ranges>

#include <EngineInterface/SpaceCalculator.h>

#include <EngineGpuKernels/DebugKernels.cuh>
#include <EngineGpuKernels/ForceFieldKernels.cuh>
#include <EngineGpuKernels/SimulationKernels.cuh>
#include <EngineGpuKernels/SimulationStatistics.cuh>

#include "GarbageCollectorKernelsService.cuh"

void SimulationKernelsService::init()
{
    CHECK_FOR_CUDA_ERROR(cudaStreamCreate(&_stream));
}

void SimulationKernelsService::shutdown()
{
    // Destroy all cached graph executables
    for (cudaGraphExec_t& graphExec : _graphCache | std::views::values) {
        CHECK_FOR_CUDA_ERROR(cudaGraphExecDestroy(graphExec));
    }
    _graphCache.clear();

    // Destroy all cached preview graph executables
    for (cudaGraphExec_t& graphExec : _previewGraphCache | std::views::values) {
        CHECK_FOR_CUDA_ERROR(cudaGraphExecDestroy(graphExec));
    }
    _previewGraphCache.clear();

    if (_stream) {
        CHECK_FOR_CUDA_ERROR(cudaStreamDestroy(_stream));
        _stream = nullptr;
    }
}

int SimulationKernelsService::calcOptimalThreadsForFluidKernel(SimulationParameters const& parameters) const
{
    auto scanRectLength = ceilf(parameters.smoothingLength.value * 2) * 2 + 1;
    return static_cast<int>(scanRectLength * scanRectLength);
}

CudaGraphConfig SimulationKernelsService::buildGraphConfig(SettingsForSimulation const& settings, SimulationData const& data, int counterMod3) const
{
    CudaGraphConfig config;
    config.counterMod3 = counterMod3 % 3;
    config.motionType = settings.simulationParameters.motionType.value;
    config.hasLayers = settings.simulationParameters.numLayers > 0;
    config.constructorCheck = settings.simulationParameters.constructorCompletenessCheck.value;
    config.rigidityEnabled = isRigidityUpdateEnabled(settings);
    config.fluidKernelThreads = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
    config.numBlocks = settings.cudaSettings.numBlocks;
    return config;
}

void SimulationKernelsService::launchTimestepKernels(
    CudaGraphConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    auto numBlocks = config.numBlocks;
    bool calcAngularForces = (config.counterMod3 == 0);
    bool considerInnerFriction = (config.counterMod3 == 0);
    bool considerRigidityUpdate = (config.counterMod3 == 0);

    STREAM_KERNEL_CALL_1_1(cudaNextTimestep_prepare, _stream, data);

    STREAM_KERNEL_CALL(cudaNextTimestep_physics_init, _stream, numBlocks, data);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, _stream, numBlocks, 64, data);

    if (config.motionType == MotionType_Fluid) {
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, _stream, numBlocks, config.fluidKernelThreads, data);
    } else {
        STREAM_KERNEL_CALL(cudaNextTimestep_physics_calcCollisionForces, _stream, numBlocks, data);
    }

    if (config.hasLayers) {
        STREAM_KERNEL_CALL(cudaApplyForceFieldSettings, _stream, numBlocks, data);
    }

    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, _stream, numBlocks, 16, data);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, _stream, numBlocks, 16, data, calcAngularForces);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, _stream, numBlocks, 16, data);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, _stream, numBlocks, 16, data, calcAngularForces);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, _stream, numBlocks, 16, data);

    // Signal processing
    STREAM_KERNEL_CALL(cudaNextTimestep_signal_calcFutureSignals, _stream, numBlocks, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_signal_updateSignals, _stream, numBlocks, data);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_signal_neuralNetworks, _stream, numBlocks, MAX_CHANNELS * MAX_CHANNELS, data, statistics);

    // Energy flow
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_energyFlow, _stream, numBlocks, 32, data);

    // Cell type-specific functions
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep1, _stream, numBlocks, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep2, _stream, numBlocks, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_generator, _stream, numBlocks, data, statistics);

    if (config.constructorCheck) {
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_constructor_completenessCheck, _stream, numBlocks, data, statistics);
    }

    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_constructor, _stream, numBlocks, 4, data, statistics, false);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_injector, _stream, numBlocks, data, statistics);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_cellType_attacker, _stream, numBlocks, 4, data, statistics);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_cellType_depot, _stream, numBlocks, 4, data, statistics);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_muscle, _stream, numBlocks, data, statistics);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_cellType_sensor, _stream, numBlocks, 64, data, statistics);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_reconnector, _stream, numBlocks, data, statistics);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_detonator, _stream, numBlocks, data, statistics);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_digestor, _stream, numBlocks, data, statistics);
    STREAM_KERNEL_CALL(cudaNextTimestep_cellType_memory, _stream, numBlocks, data, statistics);
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_cellType_communicator, _stream, numBlocks, 64, data, statistics);

    if (considerInnerFriction) {
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, _stream, numBlocks, 16, data);
    }
    STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, _stream, numBlocks, 16, data);

    if (considerRigidityUpdate && config.rigidityEnabled) {
        STREAM_KERNEL_CALL(cudaInitClusterData, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaFindClusterIteration, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaFindClusterIteration, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaFindClusterIteration, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaFindClusterBoundaries, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaAccumulateClusterPosAndVel, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaAccumulateClusterAngularProp, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaApplyClusterData, _stream, numBlocks, data);
    }

    STREAM_KERNEL_CALL_1_1(cudaNextTimestep_structuralOperations_substep1, _stream, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_structuralOperations_substep2, _stream, numBlocks, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_structuralOperations_substep3, _stream, numBlocks, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_structuralOperations_substep4, _stream, numBlocks, data);
    STREAM_KERNEL_CALL(cudaNextTimestep_structuralOperations_substep5, _stream, numBlocks, data);

    STREAM_KERNEL_CALL_1_1(cudaNextTimestep_incTimestep, _stream, data);
}

cudaGraphExec_t SimulationKernelsService::captureTimestepGraph(
    CudaGraphConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    cudaGraph_t graph;

    CHECK_FOR_CUDA_ERROR(cudaStreamBeginCapture(_stream, cudaStreamCaptureModeGlobal));

    launchTimestepKernels(config, settings, data, statistics);

    CHECK_FOR_CUDA_ERROR(cudaStreamEndCapture(_stream, &graph));

    cudaGraphExec_t graphExec;
    CHECK_FOR_CUDA_ERROR(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CHECK_FOR_CUDA_ERROR(cudaGraphDestroy(graph));

    _graphCache[config] = graphExec;
    return graphExec;
}

void SimulationKernelsService::calcTimestep(SettingsForSimulation const& settings, SimulationData const& data, SimulationStatistics const& statistics)
{
    // Build configuration key for graph caching
    static int counterMod3 = 0;
    counterMod3 = ++counterMod3 % 3;
    auto config = buildGraphConfig(settings, data, counterMod3);

    // In debug mode, bypass CUDA Graphs to get precise kernel crash information
    if (GlobalSettings::get().isDebugMode()) {
        launchTimestepKernels(config, settings, data, statistics);
        CHECK_FOR_CUDA_ERROR(cudaStreamSynchronize(_stream));
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
        CHECK_FOR_CUDA_ERROR(cudaGraphLaunch(graphExec, _stream));

        // Wait for the graph to complete before garbage collection
        CHECK_FOR_CUDA_ERROR(cudaStreamSynchronize(_stream));
    }

    // Garbage collection cannot be part of the graph due to dynamic behavior
    GarbageCollectorKernelsService::get().cleanupAfterTimestep(settings.cudaSettings, data);
}

CudaGraphPreviewConfig SimulationKernelsService::buildPreviewGraphConfig(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    int counterMod3,
    bool detailSimulation) const
{
    CudaGraphPreviewConfig config;
    config.counterMod3 = counterMod3;
    config.detailSimulation = detailSimulation;
    config.fluidKernelThreads = calcOptimalThreadsForFluidKernel(settings.simulationParameters);
    config.numBlocks = settings.cudaSettings.numBlocks;
    return config;
}

void SimulationKernelsService::launchPreviewKernels(
    CudaGraphPreviewConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    auto numBlocks = config.numBlocks;
    bool considerForcesFromAngleDifferences = (config.counterMod3 == 0);
    bool considerInnerFriction = (config.counterMod3 == 0);

    if (!config.detailSimulation) {
        STREAM_KERNEL_CALL_1_1(cudaNextTimestep_prepare, _stream, data);

        STREAM_KERNEL_CALL(cudaNextTimestep_physics_init, _stream, numBlocks, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, _stream, numBlocks, 64, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, _stream, numBlocks, config.fluidKernelThreads, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, _stream, numBlocks, 16, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, _stream, numBlocks, 16, data, considerForcesFromAngleDifferences);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, _stream, numBlocks, 16, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, _stream, numBlocks, 16, data, considerForcesFromAngleDifferences);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, _stream, numBlocks, 16, data);

        // Cell type-specific functions
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep1, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep2, _stream, numBlocks, data);

        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_constructor, _stream, numBlocks, 4, data, statistics, true);

        if (considerInnerFriction) {
            STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, _stream, numBlocks, 16, data);
        }
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, _stream, numBlocks, 16, data);

        STREAM_KERNEL_CALL_1_1(cudaNextTimestep_incTimestep, _stream, data);
    } else {
        STREAM_KERNEL_CALL_1_1(cudaNextTimestep_prepare, _stream, data);

        STREAM_KERNEL_CALL(cudaNextTimestep_physics_init, _stream, numBlocks, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_fillMaps, _stream, numBlocks, 64, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcFluidForces, _stream, numBlocks, config.fluidKernelThreads, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyForces, _stream, numBlocks, 16, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, _stream, numBlocks, 16, data, considerForcesFromAngleDifferences);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_verletPositionUpdate, _stream, numBlocks, 16, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_calcConnectionForces, _stream, numBlocks, 16, data, considerForcesFromAngleDifferences);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_verletVelocityUpdate, _stream, numBlocks, 16, data);

        // Signal processing
        STREAM_KERNEL_CALL(cudaNextTimestep_signal_calcFutureSignals, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaNextTimestep_signal_updateSignals, _stream, numBlocks, data);
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_signal_neuralNetworks, _stream, numBlocks, MAX_CHANNELS * MAX_CHANNELS, data, statistics);

        // Energy flow
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_energyFlow, _stream, numBlocks, 32, data);

        // Cell type-specific functions
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep1, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_prepare_substep2, _stream, numBlocks, data);
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_generator, _stream, numBlocks, data, statistics);

        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_constructor, _stream, numBlocks, 4, data, statistics, true);
        STREAM_KERNEL_CALL(cudaNextTimestep_cellType_muscle, _stream, numBlocks, data, statistics);

        if (considerInnerFriction) {
            STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyInnerFriction, _stream, numBlocks, 16, data);
        }
        STREAM_KERNEL_CALL_MOD(cudaNextTimestep_physics_applyFriction, _stream, numBlocks, 16, data);

        STREAM_KERNEL_CALL_1_1(cudaNextTimestep_incTimestep, _stream, data);
    }
}

cudaGraphExec_t SimulationKernelsService::capturePreviewGraph(
    CudaGraphPreviewConfig const& config,
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics)
{
    cudaGraph_t graph;

    CHECK_FOR_CUDA_ERROR(cudaStreamBeginCapture(_stream, cudaStreamCaptureModeGlobal));

    launchPreviewKernels(config, settings, data, statistics);

    CHECK_FOR_CUDA_ERROR(cudaStreamEndCapture(_stream, &graph));

    cudaGraphExec_t graphExec;
    CHECK_FOR_CUDA_ERROR(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CHECK_FOR_CUDA_ERROR(cudaGraphDestroy(graph));

    _previewGraphCache[config] = graphExec;
    return graphExec;
}

void SimulationKernelsService::calcTimestepForPreview(
    SettingsForSimulation const& settings,
    SimulationData const& data,
    SimulationStatistics const& statistics,
    bool detailSimulation)
{
    // Build configuration key for graph caching
    static int counterMod3 = 0;
    counterMod3 = ++counterMod3 % 3;
    auto config = buildPreviewGraphConfig(settings, data, counterMod3, detailSimulation);

    // In debug mode, bypass CUDA Graphs to get precise kernel crash information
    if (GlobalSettings::get().isDebugMode()) {
        launchPreviewKernels(config, settings, data, statistics);
        CHECK_FOR_CUDA_ERROR(cudaStreamSynchronize(_stream));
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
        CHECK_FOR_CUDA_ERROR(cudaGraphLaunch(graphExec, _stream));

        // Wait for the graph to complete before garbage collection
        CHECK_FOR_CUDA_ERROR(cudaStreamSynchronize(_stream));
    }

    // Garbage collection cannot be part of the graph due to dynamic behavior
    if (!detailSimulation) {
        GarbageCollectorKernelsService::get().cleanupAfterTimestepForPreview(settings.cudaSettings, data);
    } else {
        GarbageCollectorKernelsService::get().cleanupAfterTimestep(settings.cudaSettings, data);
    }
}

void SimulationKernelsService::prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& data)
{
    // Invalidate graph cache when simulation parameters change
    // The cache will be rebuilt lazily on next calcTimestep call
    for (auto& pair : _graphCache) {
        CHECK_FOR_CUDA_ERROR(cudaGraphExecDestroy(pair.second));
    }
    _graphCache.clear();

    // Also invalidate preview graph cache
    for (auto& pair : _previewGraphCache) {
        CHECK_FOR_CUDA_ERROR(cudaGraphExecDestroy(pair.second));
    }
    _previewGraphCache.clear();

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
