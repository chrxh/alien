#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/ArraySizesForGpu.h>
#include <EngineInterface/ArraySizesForTO.h>
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

    ArraySizesForTO estimateCapacityNeededForTO(CudaSettings const& gpuSettings, SimulationData const& data);
    void getData(CudaSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, TO const& to);
    void getSelectedData(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters, TO const& to);
    void getInspectedData(CudaSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, TO const& to);
    void getOverlayData(CudaSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, TO const& to);
    bool getGenomeOfCreature(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t creatureId, TO const& to);

    ArraySizesForGpu estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, TO const& to);
    void addData(CudaSettings const& gpuSettings, SimulationData const& data, TO const& to, bool selectData);
    void clearData(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    DataAccessKernelsService() = default;

    // Gpu memory
    Cell** _cudaCellArray = nullptr;
    ArraySizesForGpu* _arraySizesGPU = nullptr;
    ArraySizesForTO* _arraySizesTO = nullptr;
    bool* _foundResult = nullptr;
};
