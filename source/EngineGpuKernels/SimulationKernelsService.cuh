#pragma once

#include <unordered_map>

#include <Base/Singleton.h>

#include <EngineInterface/SettingsForSimulation.h>

#include "Definitions.cuh"
#include "Macros.cuh"

// Configuration key for CUDA Graph caching
// Captures all runtime-varying parameters that affect kernel execution
struct CudaGraphConfig
{
    int counterMod3;            // Not every kernel needs to be executed each time
    int motionType;             // MotionType_Fluid or MotionType_Collision
    bool hasLayers;             // settings.simulationParameters.numLayers > 0
    bool constructorCheck;      // settings.simulationParameters.constructorCompletenessCheck.value
    bool rigidityEnabled;       // isRigidityUpdateEnabled(settings)
    int fluidKernelThreads;     // calcOptimalThreadsForFluidKernel result
    int numBlocks;              // gpuSettings.numBlocks

    bool operator==(CudaGraphConfig const& other) const = default;
    auto operator<=>(CudaGraphConfig const& other) const = default;
};

class SimulationKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationKernelsService);

public:
    void init();
    void shutdown();

    void calcTimestep(SettingsForSimulation const& settings, SimulationData const& simulationData, SimulationStatistics const& statistics);
    void calcTimestepForPreview(
        SettingsForSimulation const& settings,
        SimulationData const& simulationData,
        SimulationStatistics const& statistics,
        bool detailSimulation);
    void prepareForSimulationParametersChanges(SettingsForSimulation const& settings, SimulationData const& simulationData);

private:
    SimulationKernelsService() = default;

    bool isRigidityUpdateEnabled(SettingsForSimulation const& settings) const;
    int calcOptimalThreadsForFluidKernel(SimulationParameters const& parameters) const;

    CudaGraphConfig buildGraphConfig(SettingsForSimulation const& settings, SimulationData const& data, int counterMod3) const;

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

    cudaStream_t _stream = nullptr;
    std::map<CudaGraphConfig, cudaGraphExec_t> _graphCache;
};
