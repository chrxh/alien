#pragma once

#include "EngineInterface/CudaSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Base.cuh"
#include "Definitions.cuh"

class _EditKernelsService
{
public:
    _EditKernelsService();
    ~_EditKernelsService();

    void removeSelection(CudaSettings const& gpuSettings, SimulationData const& data);
    void swapSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void switchSelection(CudaSettings const& gpuSettings, SimulationData const& data, PointSelectionData const& switchData);
    void setSelection(CudaSettings const& gpuSettings, SimulationData const& data, AreaSelectionData const& setData);
    void updateSelection(CudaSettings const& gpuSettings, SimulationData const& data);

    void getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult);
    void shallowUpdateSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, ShallowUpdateSelectionData const& updateData);
    void removeSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void relaxSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void uniformVelocities(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void makeSticky(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void removeStickiness(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void setBarrier(CudaSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters);
    void reconnect(CudaSettings const& gpuSettings, SimulationData const& data);
    void changeSimulationData(CudaSettings const& gpuSettings, SimulationData const& data, TO const& changeDataTO);
    bool changeCreature(CudaSettings const& gpuSettings, SimulationData const& data, TO const& dataTO);   // dataTO only contains 1 genome
    void colorSelectedCells(CudaSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters);
    void setDetached(CudaSettings const& gpuSettings, SimulationData const& data, bool value);

    void applyForce(CudaSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData);

    void rolloutSelection(CudaSettings const& gpuSettings, SimulationData const& data);

    void applyCataclysm(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollector;

    // Gpu memory
    int* _cudaRolloutResult;
    int* _cudaSwitchResult;
    int* _cudaUpdateResult;
    int* _cudaRemoveResult;
    float2* _cudaCenter;
    float2* _cudaVelocity;
    int* _cudaNumEntities;
    unsigned long long int* _cudaMinCellPosYAndIndex;
    Genome** _genomePtr;
    Creature** _creaturePtr;
    bool* _result;
};
