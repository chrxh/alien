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

    void shallowUpdateSelectedObjects(KernelLaunchSettings const& launchSettings, SimulationData const& data, ShallowUpdateSelectionData const& updateData);
    void removeSelectedObjects(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters);
    void relaxSelectedObjects(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters);
    void uniformVelocities(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters);
    void makeSticky(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters);
    void removeStickiness(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters);
    void setStatic(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool value, bool includeClusters);
    void reconnect(KernelLaunchSettings const& launchSettings, SimulationData const& data);
    void glueSelectedObjects(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters);
    void cutConnections(
        KernelLaunchSettings const& launchSettings,
        SimulationData const& data,
        float2 const& cutStart,
        float2 const& cutEnd,
        bool onlySelected,
        bool includeClusters);
    void changeSimulationData(KernelLaunchSettings const& launchSettings, SimulationData const& data, TOs const& changeTO);
    int injectGenomeToSelectedCreatures(KernelLaunchSettings const& launchSettings, SimulationData const& data, TOs const& to);  // to only contains 1 genome
    void colorSelectedCells(KernelLaunchSettings const& launchSettings, SimulationData const& data, unsigned char color, bool includeClusters);
    void setDetached(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool value);

    void applyForce(KernelLaunchSettings const& launchSettings, SimulationData const& data, ApplyForceData const& applyData);

    void applyCataclysm(KernelLaunchSettings const& launchSettings, SimulationData const& data);

    void getSelectionShallowData(KernelLaunchSettings const& launchSettings, SimulationData const& data, SelectionResult const& selectionResult);

private:
    EditKernelsService() = default;

    void disconnectOverstretchedConnections(KernelLaunchSettings const& launchSettings, SimulationData const& data);
    void connectSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters, bool onlyWithinSelection);

    float2 flattenSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data);

    // Gpu memory
    int* _cudaRolloutResult = nullptr;
    int* _cudaSwitchResult = nullptr;
    int* _cudaUpdateResult = nullptr;
    int* _cudaRemoveResult = nullptr;
    int* _cudaInjectResult = nullptr;
    float2* _cudaCenter = nullptr;
    float2* _cudaVelocity = nullptr;
    int* _cudaNumEntities = nullptr;
    float4* _cudaAngleSums = nullptr;
    Genome** _genomePtr = nullptr;
};
