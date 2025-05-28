#pragma once

#include "EngineInterface/ArraySizesForTO.h"
#include "EngineInterface/ArraySizesForGpu.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/InspectedEntityIds.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Macros.cuh"

class _DataAccessKernelsService
{
public:
    _DataAccessKernelsService();
    ~_DataAccessKernelsService();

    ArraySizesForTO estimateCapacityNeededForTO(GpuSettings const& gpuSettings, SimulationData const& data);
    void getData(GpuSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, CollectionTO const& dataTO);
    void getSelectedData(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters, CollectionTO const& dataTO);
    void getInspectedData(GpuSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, CollectionTO const& dataTO);
    void getOverlayData(GpuSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, CollectionTO const& dataTO);

    ArraySizesForGpu estimateCapacityNeededForGpu(GpuSettings const& gpuSettings, CollectionTO const& dataTO);
    void addData(GpuSettings const& gpuSettings, SimulationData const& data, CollectionTO const& dataTO, bool selectData);
    void clearData(GpuSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollectorKernels;
    EditKernelsService _editKernels;

    // Gpu memory
    Cell** _cudaCellArray;
    ArraySizesForGpu* _arraySizesGPU;
    ArraySizesForTO* _arraySizesTO;
};

