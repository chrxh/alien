#pragma once

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include "Base.cuh"
#include "Definitions.cuh"

class _SelectionKernelsService
{
public:
    _SelectionKernelsService();
    ~_SelectionKernelsService();

    void removeSelection(CudaSettings const& gpuSettings, SimulationData const& data);
    void swapSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void switchSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void setSelection(CudaSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData);
    void updateSelection(CudaSettings const& gpuSettings, SimulationData const& data);

    void getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult);

    void rolloutSelection(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollector;

    // Gpu memory
    int* _cudaRolloutResult;
    int* _cudaSwitchResult;
    unsigned long long int* _cudaMinCellPosYAndIndex;
};
