#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/ArraySizesForGpuEntities.h>
#include <EngineInterface/ArraySizesForTOs.h>
#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/KernelLaunchSettings.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include <EngineKernels/Base.cuh>
#include <EngineKernels/Definitions.cuh>
#include <EngineKernels/Macros.cuh>

class DataAccessKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DataAccessKernelsService);

public:
    void init();
    void shutdown();

    ArraySizesForTOs estimateCapacityNeededForTO(KernelLaunchSettings const& launchSettings, SimulationData const& data);
    void getData(KernelLaunchSettings const& launchSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, TOs const& to);
    void getSelectedData(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters, TOs const& to);
    void getInspectedData(KernelLaunchSettings const& launchSettings, SimulationData const& data, InspectedEntityIds entityIds, TOs const& to);
    void getOverlayData(KernelLaunchSettings const& launchSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, TOs const& to);

    ArraySizesForGpuEntities estimateCapacityNeededForGpu(KernelLaunchSettings const& launchSettings, TOs const& to);
    void addData(KernelLaunchSettings const& launchSettings, SimulationData const& data, TOs const& to, bool selectData);
    void clearData(KernelLaunchSettings const& launchSettings, SimulationData const& data);

private:
    DataAccessKernelsService() = default;

    // Gpu memory
    Object** _cudaCellArray = nullptr;
    ArraySizesForGpuEntities* _arraySizesGPU = nullptr;
    ArraySizesForTOs* _arraySizesTO = nullptr;
    bool* _foundResult = nullptr;
};
