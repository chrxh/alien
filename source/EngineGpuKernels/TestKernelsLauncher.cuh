#pragma once

#include "Definitions.cuh"
#include "EngineInterface/GpuSettings.h"

class _TestKernelsLauncher
{
public:
    void testOnly_mutateCellFunction(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
};
