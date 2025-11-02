#pragma once

#include <EngineInterface/CudaSettings.h>

#include "Base.cuh"
#include "Definitions.cuh"
#include "Macros.cuh"

class _StatisticsKernelsService
{
public:
    void updateStatistics(CudaSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

private:
};
