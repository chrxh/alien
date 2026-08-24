#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/KernelLaunchSettings.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/Definitions.cuh>

class SelectionKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SelectionKernelsService);

public:
    void init();
    void shutdown();

    void removeSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data);
    void swapSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, PointSelectionData const& switchData);
    void switchSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, PointSelectionData const& switchData);
    void setSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data, AreaSelectionData const& setData);
    void updateSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data);

    void rolloutSelection(KernelLaunchSettings const& launchSettings, SimulationData const& data);

private:
    SelectionKernelsService() = default;

    // Gpu memory
    int* _cudaRolloutResult = nullptr;
    int* _cudaSwitchResult = nullptr;
};
