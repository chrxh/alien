#pragma once

#include <unordered_map>

#include <Base/Singleton.h>

#include <EngineInterface/SettingsForSimulation.h>

#include <EngineKernels/Definitions.cuh>
#include <EngineKernels/Macros.cuh>

// Configuration key for CUDA Graph caching
// Captures all runtime-varying parameters that affect kernel execution
struct CudaGraphConfig
{
    int domainIndex;           // The captured kernel parameters point to the data of one domain
    bool isDecomposed;         // Requests of other domains need to be processed
    int timestepMod3;          // Not every kernel needs to be executed each time
    bool executeCellFunction;  // Cell type functions need to be executed
    bool hasLayers;            // settings.simulationParameters.numLayers > 0
    bool rigidityEnabled;      // isRigidityUpdateEnabled(settings)
    int numBlocks;             // settings.kernelLaunchSettings.numBlocks

    bool operator==(CudaGraphConfig const& other) const = default;
    auto operator<=>(CudaGraphConfig const& other) const = default;
};

// Configuration key for Preview CUDA Graph caching
struct CudaGraphPreviewConfig
{
    int timestepMod3;           // Not every kernel needs to be executed each time
    bool executeCellFunctions;  // Cell type functions need to be executed each TIMESTEPS_PER_CELL_FUNCTION
    bool detailSimulation;      // Whether detail simulation is enabled
    int numBlocks;              // settings.kernelLaunchSettings.numBlocks

    bool operator==(CudaGraphPreviewConfig const& other) const = default;
    auto operator<=>(CudaGraphPreviewConfig const& other) const = default;
};

class SimulationKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationKernelsService);

public:
    void init();
    void shutdown();

    // A time step is launched and finished separately, so that the domains on several devices can compute it at the same time
    void launchTimestep(
        SettingsForSimulation const& settings,
        SimulationData const& simulationData,
        SimulationStatistics const& statistics,
        uint64_t timestep,
        bool forceCellFunctionExecution);
    void finishTimestep(SettingsForSimulation const& settings, SimulationData const& simulationData);
    void calcTimestepForPreview(
        SettingsForSimulation const& settings,
        SimulationData const& simulationData,
        SimulationStatistics const& statistics,
        uint64_t timestep,
        bool forceCellFunctionExecution,
        bool detailSimulation);
    void prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& simulationData);

private:
    SimulationKernelsService() = default;

    static auto constexpr PreviewStreamKey = -1;

    // Each domain gets its own stream so that domains sharing a device can overlap
    cudaStream_t getStream(int key);

    bool isRigidityUpdateEnabled(SettingsForSimulation const& settings) const;

    CudaGraphConfig buildGraphConfig(SettingsForSimulation const& settings, SimulationData const& data, uint64_t timestep, bool forceCellFunctionExecution)
        const;

    cudaGraphExec_t captureTimestepGraph(
        CudaGraphConfig const& config,
        SettingsForSimulation const& settings,
        SimulationData const& data,
        SimulationStatistics const& statistics);

    void launchTimestepKernels(
        CudaGraphConfig const& config,
        SettingsForSimulation const& settings,
        SimulationData const& data,
        SimulationStatistics const& statistics);

    CudaGraphPreviewConfig buildPreviewGraphConfig(
        SettingsForSimulation const& settings,
        SimulationData const& data,
        uint64_t timestep,
        bool forceCellFunctionExecution,
        bool detailSimulation) const;

    cudaGraphExec_t capturePreviewGraph(
        CudaGraphPreviewConfig const& config,
        SettingsForSimulation const& settings,
        SimulationData const& data,
        SimulationStatistics const& statistics);

    void launchPreviewKernels(
        CudaGraphPreviewConfig const& config,
        SettingsForSimulation const& settings,
        SimulationData const& data,
        SimulationStatistics const& statistics);

    struct Stream
    {
        cudaStream_t stream;
        int device;
    };
    std::map<int, Stream> _streams;
    std::map<CudaGraphConfig, cudaGraphExec_t> _graphCache;
    std::map<CudaGraphPreviewConfig, cudaGraphExec_t> _previewGraphCache;
};
