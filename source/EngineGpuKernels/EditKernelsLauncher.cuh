#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Base.cuh"
#include "Definitions.cuh"

class _EditKernelsLauncher
{
public:
    _EditKernelsLauncher();
    ~_EditKernelsLauncher();

    void removeSelection(GpuSettings const& gpuSettings, SimulationData const& data);
    void swapSelection(GpuSettings const& gpuSettings, SimulationData const& data, PointSelectionData switchData);
    void switchSelection(GpuSettings const& gpuSettings, SimulationData data, PointSelectionData switchData);
    void setSelection(GpuSettings const& gpuSettings, SimulationData data, AreaSelectionData setData);
    void updateSelection(GpuSettings const& gpuSettings, SimulationData data);
    void shallowUpdateSelectedEntities(GpuSettings const& gpuSettings, SimulationData data, ShallowUpdateSelectionData updateData);

    void rolloutSelection(GpuSettings const& gpuSettings, SimulationData data);

private:
    int* _cudaRolloutResult;
    int* _cudaSwitchResult;
    int* _cudaUpdateResult;
    float2* _cudaCenter;
    int* _cudaNumEntities;

    GarbageCollectorKernelsLauncher _garbageCollector;
};
