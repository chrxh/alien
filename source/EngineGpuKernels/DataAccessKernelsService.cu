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
    TO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaPrepareCreaturesAndGenomesForConversionToTO, rectUpperLeft, rectLowerRight, data);
    KERNEL_CALL(cudaGetGenomeData, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaGetCreatureData, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaGetCellDataWithoutConnections, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, data, dataTO);
}

void _DataAccessKernelsService::getSelectedData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    bool includeClusters,
    TO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaPrepareSelectedCreaturesForConversionToTO, includeClusters, data);
    KERNEL_CALL(cudaGetSelectedGenomeData, data, includeClusters, dataTO);
    KERNEL_CALL(cudaGetSelectedCreatureData, data, includeClusters, dataTO);
    KERNEL_CALL(cudaGetSelectedCellDataWithoutConnections, data, includeClusters, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetSelectedParticleData, data, dataTO);
}

void _DataAccessKernelsService::getInspectedData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    InspectedEntityIds entityIds,
    TO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaPrepareCreaturesAndGenomesForConversionToTO, entityIds, data);
    KERNEL_CALL(cudaGetGenomeData, entityIds, data, dataTO);
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
    TO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetOverlayData, rectUpperLeft, rectLowerRight, data, dataTO);
}

ArraySizesForGpu _DataAccessKernelsService::estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, TO const& dataTO)
{
    setValueToDevice(_arraySizesGPU, ArraySizesForGpu{});
    KERNEL_CALL(cudaEstimateCapacityNeededForGpu, dataTO, _arraySizesGPU);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesGPU);
}

void _DataAccessKernelsService::addData(CudaSettings const& gpuSettings, SimulationData const& data, TO const& dataTO, bool selectData)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);
    KERNEL_CALL(cudaAdaptNumberGenerator, data.primaryNumberGen, dataTO);

    KERNEL_CALL_1_1(cudaGetArraysBasedOnTO, data, dataTO, _cudaCellArray);
    KERNEL_CALL(cudaSetGenomeDataFromTO, data, dataTO);
    KERNEL_CALL(cudaSetCreatureDataFromTO, data, dataTO);
    KERNEL_CALL(cudaSetCellAndParticleDataFromTO, data, dataTO, _cudaCellArray, selectData);
    _garbageCollectorKernels->cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        _editKernels->rolloutSelection(gpuSettings, data);
    }
}

void _DataAccessKernelsService::clearData(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaClearData, data);
}
