#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/Definitions.cuh>

class EditKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EditKernelsService);

public:
    void init();
    void shutdown();

    void shallowUpdateSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, ShallowUpdateSelectionData const& updateData);
    void removeSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void relaxSelectedObjects(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void uniformVelocities(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void makeSticky(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void removeStickiness(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void setBarrier(CudaSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters);
    void reconnect(CudaSettings const& gpuSettings, SimulationData const& data);
    void changeSimulationData(CudaSettings const& gpuSettings, SimulationData const& data, TOs const& changeTO);
    int injectGenomeToSelectedCreatures(CudaSettings const& gpuSettings, SimulationData const& data, TOs const& to);  // to only contains 1 genome
    void colorSelectedCells(CudaSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters);
    void setDetached(CudaSettings const& gpuSettings, SimulationData const& data, bool value);

    void applyForce(CudaSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData);

    void applyCataclysm(CudaSettings const& gpuSettings, SimulationData const& data);

    void getSelectionShallowData(CudaSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult);

private:
    EditKernelsService() = default;

    // Gpu memory
    int* _cudaRolloutResult = nullptr;
    int* _cudaSwitchResult = nullptr;
    int* _cudaUpdateResult = nullptr;
    int* _cudaRemoveResult = nullptr;
    int* _cudaInjectResult = nullptr;
    float2* _cudaCenter = nullptr;
    float2* _cudaVelocity = nullptr;
    int* _cudaNumEntities = nullptr;
    unsigned long long int* _cudaMinCellPosYAndIndex = nullptr;
    Genome** _genomePtr = nullptr;
};
