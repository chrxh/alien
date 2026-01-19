#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/ArraySizesForGpuEntities.h>
#include <EngineInterface/ArraySizesForTOs.h>
#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>

#include <EngineGpuKernels/Base.cuh>
#include <EngineGpuKernels/Definitions.cuh>
#include <EngineGpuKernels/Macros.cuh>

class DataAccessKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(DataAccessKernelsService);

public:
    void init();
    void shutdown();

    ArraySizesForTOs estimateCapacityNeededForTO(CudaSettings const& gpuSettings, SimulationData const& data);
    void getData(CudaSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, TOs const& to);
    void getSelectedData(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters, TOs const& to);
    void getInspectedData(CudaSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, TOs const& to);
    void getOverlayData(CudaSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, TOs const& to);
    bool getGenomeOfCreature(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t creatureId, TOs const& to);

    ArraySizesForGpuEntities estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, TOs const& to);
    void addData(CudaSettings const& gpuSettings, SimulationData const& data, TOs const& to, bool selectData);
    void clearData(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    DataAccessKernelsService() = default;

    // Gpu memory
    Object** _cudaCellArray = nullptr;
    ArraySizesForGpuEntities* _arraySizesGPU = nullptr;
    ArraySizesForTOs* _arraySizesTO = nullptr;
    bool* _foundResult = nullptr;
};
