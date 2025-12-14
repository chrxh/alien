#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include "Base.cuh"
#include "Definitions.cuh"

class SelectionKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SelectionKernelsService);

public:
    void init();
    void shutdown();

    void removeSelection(CudaSettings const& gpuSettings, SimulationData const& data);
    void swapSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void switchSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void setSelection(CudaSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData);
    void updateSelection(CudaSettings const& gpuSettings, SimulationData const& data);

    void getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult);

    void rolloutSelection(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    SelectionKernelsService() = default;

    // Gpu memory
    int* _cudaRolloutResult = nullptr;
    int* _cudaSwitchResult = nullptr;
    unsigned long long int* _cudaMinCellPosYAndIndex = nullptr;
};
