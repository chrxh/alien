#pragma once

#include <Base/Interface/Singleton.h>

#include <Engine/Interface/ArraySizesForGpuEntities.h>
#include <Engine/Interface/ArraySizesForTOs.h>
#include <Engine/Interface/InspectedEntityIds.h>
#include <Engine/Interface/KernelLaunchSettings.h>
#include <Engine/Interface/ShallowUpdateSelectionData.h>

#include <Engine/Kernels/Base.cuh>
#include <Engine/Kernels/Definitions.cuh>
#include <Engine/Kernels/Macros.cuh>

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
