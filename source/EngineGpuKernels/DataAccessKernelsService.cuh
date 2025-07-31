#pragma once

#include <cuda_runtime.h>

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
    void getData(CudaSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, CollectionTO const& dataTO);
    void getData(CudaSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, CollectionTO const& dataTO, cudaStream_t stream);
    void getSelectedData(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters, CollectionTO const& dataTO);
    void getInspectedData(CudaSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, CollectionTO const& dataTO);
    void getOverlayData(CudaSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, CollectionTO const& dataTO);

    ArraySizesForGpu estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, CollectionTO const& dataTO);
    void addData(CudaSettings const& gpuSettings, SimulationData const& data, CollectionTO const& dataTO, bool selectData);
    void addData(CudaSettings const& gpuSettings, SimulationData const& data, CollectionTO const& dataTO, bool selectData, cudaStream_t stream);
    void clearData(CudaSettings const& gpuSettings, SimulationData const& data);
    void clearData(CudaSettings const& gpuSettings, SimulationData const& data, cudaStream_t stream);

private:
    GarbageCollectorKernelsService _garbageCollectorKernels;
    EditKernelsService _editKernels;

    // Gpu memory
    Cell** _cudaCellArray;
    ArraySizesForGpu* _arraySizesGPU;
    ArraySizesForTO* _arraySizesTO;
};

