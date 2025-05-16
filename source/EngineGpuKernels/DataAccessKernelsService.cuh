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
    void getData(GpuSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, DataTO const& dataTO);
    void getSelectedData(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters, DataTO const& dataTO);
    void getInspectedData(GpuSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, DataTO const& dataTO);
    void getOverlayData(GpuSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, DataTO const& dataTO);

    ArraySizesForGpu estimateCapacityNeededForGpu(GpuSettings const& gpuSettings, DataTO const& dataTO);
    void addData(GpuSettings const& gpuSettings, SimulationData const& data, DataTO const& dataTO, bool selectData, bool createIds);
    void clearData(GpuSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollectorKernels;
    EditKernelsService _editKernels;

    // Gpu memory
    Cell** _cudaCellArray;
    ArraySizesForGpu* _arraySizes;
    ArraySizesForTO* _arraySizesTO;
};

