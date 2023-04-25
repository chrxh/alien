#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Macros.cuh"

class _StatisticsKernelsLauncher
{
public:
    void updateStatistics(GpuSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

private:
};
