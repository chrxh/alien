#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>

struct SimulationData;
class SimulationStatistics;

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
