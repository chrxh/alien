#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

#include <EngineGpuKernels/Base.cuh>
#include <EngineGpuKernels/Definitions.cuh>
#include <EngineGpuKernels/Macros.cuh>

class StatisticsKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(StatisticsKernelsService);

public:
    void init();
    void shutdown();

    void updateStatistics(CudaSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

private:
    StatisticsKernelsService() = default;
};
