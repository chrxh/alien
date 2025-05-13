#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Base.cuh"
#include "Definitions.cuh"

class _EditKernelsService
{
public:
    _EditKernelsService();
    ~_EditKernelsService();

    void removeSelection(GpuSettings const& gpuSettings, SimulationData const& data);
    void swapSelection(GpuSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void switchSelection(GpuSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void setSelection(GpuSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData);
    void updateSelection(GpuSettings const& gpuSettings, SimulationData const& data);

    void getSelectionShallowData(GpuSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult);
    void shallowUpdateSelectedObjects(
        GpuSettings const& gpuSettings,
        SimulationData const& data,
        ShallowUpdateSelectionData const& updateData);
    void removeSelectedObjects(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void relaxSelectedObjects(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void uniformVelocities(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void makeSticky(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void removeStickiness(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void setBarrier(GpuSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters);
    void reconnect(GpuSettings const& gpuSettings, SimulationData const& data);
    void changeSimulationData(GpuSettings const& gpuSettings, SimulationData const& data, DataTO const& changeDataTO);
    void colorSelectedCells(GpuSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters);
    void setDetached(GpuSettings const& gpuSettings, SimulationData const& data, bool value);

    void applyForce(GpuSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData);

    void rolloutSelection(GpuSettings const& gpuSettings, SimulationData const& data);

    void applyCataclysm(GpuSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollector;

    //gpu memory
    int* _cudaRolloutResult;
    int* _cudaSwitchResult;
    int* _cudaUpdateResult;
    int* _cudaRemoveResult;
    float2* _cudaCenter;
    float2* _cudaVelocity;
    int* _cudaNumEntities;
    unsigned long long int* _cudaMinCellPosYAndIndex;
};
