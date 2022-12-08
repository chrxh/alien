#pragma once

#include "Definitions.cuh"
#include "EngineInterface/GpuSettings.h"

class _TestKernelsLauncher
{
public:
    void testOnly_mutateNeuron(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
    void testOnly_mutateCellFunctionData(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
    void testOnly_mutateCellFunction(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
    void testOnly_mutateInsert(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
    void testOnly_mutateDelete(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
};
