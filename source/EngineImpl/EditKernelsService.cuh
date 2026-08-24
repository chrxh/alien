#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/KernelLaunchSettings.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/Definitions.cuh>

class EditKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EditKernelsService);

public:
    void init();
    void shutdown();

    void shallowUpdateSelectedObjects(KernelLaunchSettings const& gpuSettings, SimulationData const& data, ShallowUpdateSelectionData const& updateData);
    void removeSelectedObjects(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void relaxSelectedObjects(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void uniformVelocities(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void makeSticky(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void removeStickiness(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters);
    void setBarrier(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool value, bool includeClusters);
    void reconnect(KernelLaunchSettings const& gpuSettings, SimulationData const& data);
    void changeSimulationData(KernelLaunchSettings const& gpuSettings, SimulationData const& data, TOs const& changeTO);
    int injectGenomeToSelectedCreatures(KernelLaunchSettings const& gpuSettings, SimulationData const& data, TOs const& to);  // to only contains 1 genome
    void colorSelectedCells(KernelLaunchSettings const& gpuSettings, SimulationData const& data, unsigned char color, bool includeClusters);
    void setDetached(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool value);

    void applyForce(KernelLaunchSettings const& gpuSettings, SimulationData const& data, ApplyForceData const& applyData);

    void applyCataclysm(KernelLaunchSettings const& gpuSettings, SimulationData const& data);

    void getSelectionShallowData(KernelLaunchSettings const& gpuSettings, SimulationData const& data, SelectionResult const& selectionResult);

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
