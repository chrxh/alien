#pragma once

#include "EngineInterface/ArraySizesForTO.h"
#include "EngineInterface/ArraySizesForGpu.h"
#include "EngineInterface/CudaSettings.h"
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

    ArraySizesForTO estimateCapacityNeededForTO(CudaSettings const& gpuSettings, SimulationData const& data);
    void getData(CudaSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, TO const& to);
    void getSelectedData(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters, TO const& to);
    void getInspectedData(CudaSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, TO const& to);
    void getOverlayData(CudaSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, TO const& to);

    ArraySizesForGpu estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, TO const& to);
    void addData(CudaSettings const& gpuSettings, SimulationData const& data, TO const& to, bool selectData);
    void clearData(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollectorKernels;
    EditKernelsService _editKernels;

    // Gpu memory
    Cell** _cudaCellArray;
    ArraySizesForGpu* _arraySizesGPU;
    ArraySizesForTO* _arraySizesTO;
};

