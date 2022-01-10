#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Macros.cuh"

class _MonitorKernelsLauncher
{
public:
    void getMonitorData(GpuSettings const& gpuSettings, SimulationData const& data, CudaMonitorData const& monitorData);

private:
};
