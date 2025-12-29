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
    int timestepMod3;           // data.timestep % 3 determines which optional kernels to run
    int motionType;             // MotionType_Fluid or MotionType_Collision
    bool hasLayers;             // settings.simulationParameters.numLayers > 0
    bool constructorCheck;      // settings.simulationParameters.constructorCompletenessCheck.value
    bool rigidityEnabled;       // isRigidityUpdateEnabled(settings)
    int fluidKernelThreads;     // calcOptimalThreadsForFluidKernel result
    int numBlocks;              // gpuSettings.numBlocks

    bool operator==(CudaGraphConfig const& other) const = default;
};

struct CudaGraphConfigHash
{
    size_t operator()(CudaGraphConfig const& config) const
    {
        size_t h = 0;
        h ^= std::hash<int>()(config.timestepMod3) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(config.motionType) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(config.hasLayers) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(config.constructorCheck) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(config.rigidityEnabled) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(config.fluidKernelThreads) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(config.numBlocks) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
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

    CudaGraphConfig buildGraphConfig(
        SettingsForSimulation const& settings,
        SimulationData const& data) const;

    void captureTimestepGraph(
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
    std::unordered_map<CudaGraphConfig, cudaGraphExec_t, CudaGraphConfigHash> _graphCache;
};
