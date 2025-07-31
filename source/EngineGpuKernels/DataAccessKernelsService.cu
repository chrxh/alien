#include "DataAccessKernelsService.cuh"

#include "DataAccessKernels.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "EditKernelsService.cuh"
#include "DebugKernels.cuh"

_DataAccessKernelsService::_DataAccessKernelsService()
{
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsService>();
    _editKernels = std::make_shared<_EditKernelsService>();

    CudaMemoryManager::getInstance().acquireMemory(1, _cudaCellArray);
    CudaMemoryManager::getInstance().acquireMemory(1, _arraySizesGPU);
    CudaMemoryManager::getInstance().acquireMemory(1, _arraySizesTO);
}

_DataAccessKernelsService::~_DataAccessKernelsService()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaCellArray);
    CudaMemoryManager::getInstance().freeMemory(_arraySizesGPU);
    CudaMemoryManager::getInstance().freeMemory(_arraySizesTO);
}

ArraySizesForTO _DataAccessKernelsService::estimateCapacityNeededForTO(CudaSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_arraySizesTO, ArraySizesForTO{});
    KERNEL_CALL(cudaEstimateCapacityNeededForTO, data, _arraySizesTO);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesTO);
}

void _DataAccessKernelsService::getData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    CollectionTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaPrepareCreaturesForConversionToTO, rectUpperLeft, rectLowerRight, data);
    KERNEL_CALL(cudaGetCreatureData, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaGetCellDataWithoutConnections, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, data, dataTO);
}

void _DataAccessKernelsService::getSelectedData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    bool includeClusters,
    CollectionTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaPrepareSelectedCreaturesForConversionToTO, includeClusters, data);
    KERNEL_CALL(cudaGetSelectedCreatureData, data, includeClusters, dataTO);
    KERNEL_CALL(cudaGetSelectedCellDataWithoutConnections, data, includeClusters, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetSelectedParticleData, data, dataTO);
}

void _DataAccessKernelsService::getInspectedData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    InspectedEntityIds entityIds,
    CollectionTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaPrepareCreaturesForConversionToTO, entityIds, data);
    KERNEL_CALL(cudaGetCreatureData, entityIds, data, dataTO);
    KERNEL_CALL(cudaGetInspectedCellDataWithoutConnections, entityIds, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetInspectedParticleData, entityIds, data, dataTO);
}

void _DataAccessKernelsService::getOverlayData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    CollectionTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetOverlayData, rectUpperLeft, rectLowerRight, data, dataTO);
}

ArraySizesForGpu _DataAccessKernelsService::estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, CollectionTO const& dataTO)
{
    setValueToDevice(_arraySizesGPU, ArraySizesForGpu{});
    KERNEL_CALL(cudaEstimateCapacityNeededForGpu, dataTO, _arraySizesGPU);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesGPU);
}

void _DataAccessKernelsService::addData(CudaSettings const& gpuSettings, SimulationData const& data, CollectionTO const& dataTO, bool selectData)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);
    KERNEL_CALL(cudaAdaptNumberGenerator, data.primaryNumberGen, dataTO);

    KERNEL_CALL_1_1(cudaGetArraysBasedOnTO, data, dataTO, _cudaCellArray);
    KERNEL_CALL(cudaSetCreatureDataFromTO, data, dataTO);
    KERNEL_CALL(cudaSetDataFromTO, data, dataTO, _cudaCellArray, selectData);
    _garbageCollectorKernels->cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        _editKernels->rolloutSelection(gpuSettings, data);
    }
    KERNEL_CALL(cudaAdaptNumberGenerator, data.primaryNumberGen, dataTO);
}

void _DataAccessKernelsService::clearData(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaClearData, data);
}

void _DataAccessKernelsService::getData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    CollectionTO const& dataTO,
    cudaStream_t stream)
{
    KERNEL_CALL_1_1_STREAM(cudaClearDataTO, stream, dataTO);
    KERNEL_CALL_STREAM(cudaPrepareCreaturesForConversionToTO, stream, rectUpperLeft, rectLowerRight, data);
    KERNEL_CALL_STREAM(cudaGetCreatureData, stream, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL_STREAM(cudaGetCellDataWithoutConnections, stream, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL_STREAM(cudaResolveConnections, stream, data, dataTO);
    KERNEL_CALL_STREAM(cudaGetParticleData, stream, rectUpperLeft, rectLowerRight, data, dataTO);
}

void _DataAccessKernelsService::addData(CudaSettings const& gpuSettings, SimulationData const& data, CollectionTO const& dataTO, bool selectData, cudaStream_t stream)
{
    KERNEL_CALL_1_1_STREAM(cudaSaveNumEntries, stream, data);
    KERNEL_CALL_STREAM(cudaAdaptNumberGenerator, stream, data.primaryNumberGen, dataTO);

    KERNEL_CALL_1_1_STREAM(cudaGetArraysBasedOnTO, stream, data, dataTO, _cudaCellArray);
    KERNEL_CALL_STREAM(cudaSetCreatureDataFromTO, stream, data, dataTO);
    KERNEL_CALL_STREAM(cudaSetDataFromTO, stream, data, dataTO, _cudaCellArray, selectData);
    _garbageCollectorKernels->cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        _editKernels->rolloutSelection(gpuSettings, data, stream);
    }
    KERNEL_CALL_STREAM(cudaAdaptNumberGenerator, stream, data.primaryNumberGen, dataTO);
}

void _DataAccessKernelsService::clearData(CudaSettings const& gpuSettings, SimulationData const& data, cudaStream_t stream)
{
    KERNEL_CALL_STREAM(cudaClearData, stream, data);
}
