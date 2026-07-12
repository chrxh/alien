#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/Definitions.cuh>
#include <EngineKernels/Macros.cuh>

class StatisticsKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(StatisticsKernelsService);

public:
    void init();
    void shutdown();

    void updateStatistics(CudaSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

    // Runs at deterministic timesteps (not wall-clock throttled): heavy kernels here would otherwise
    // perturb the simulation's execution timing at run-to-run varying timesteps.
    void updateEvolutionStatistics(CudaSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

private:
    StatisticsKernelsService() = default;
};
