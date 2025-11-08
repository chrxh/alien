#include "DataAccessKernels.cuh"
#include "DataAccessKernelsService.cuh"
#include "DebugKernels.cuh"
#include "EditKernelsService.cuh"
#include "GarbageCollectorKernelsService.cuh"

_DataAccessKernelsService::_DataAccessKernelsService()
{
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsService>();
    _editKernels = std::make_shared<_EditKernelsService>();

    CudaMemoryManager::getInstance().acquireMemory(1, _cudaCellArray);
    CudaMemoryManager::getInstance().acquireMemory(1, _arraySizesGPU);
    CudaMemoryManager::getInstance().acquireMemory(1, _arraySizesTO);
    CudaMemoryManager::getInstance().acquireMemory(1, _foundResult);
}

_DataAccessKernelsService::~_DataAccessKernelsService()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaCellArray);
    CudaMemoryManager::getInstance().freeMemory(_arraySizesGPU);
    CudaMemoryManager::getInstance().freeMemory(_arraySizesTO);
    CudaMemoryManager::getInstance().freeMemory(_foundResult);
}

ArraySizesForTO _DataAccessKernelsService::estimateCapacityNeededForTO(CudaSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_arraySizesTO, ArraySizesForTO{});
    KERNEL_CALL(cudaEstimateCapacityNeededForTO_step1, data);
    KERNEL_CALL(cudaEstimateCapacityNeededForTO_step2, data, _arraySizesTO);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesTO);
}

void _DataAccessKernelsService::getData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    TO const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareCreaturesAndGenomesForConversionToTO, rectUpperLeft, rectLowerRight, data);
    KERNEL_CALL(cudaGetGenomeData, rectUpperLeft, rectLowerRight, data, to);
    KERNEL_CALL(cudaGetCreatureData, rectUpperLeft, rectLowerRight, data, to);
    KERNEL_CALL(cudaGetCellDataWithoutConnections, rectUpperLeft, rectLowerRight, data, to);
    KERNEL_CALL(cudaResolveConnections, data, to);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, data, to);
}

void _DataAccessKernelsService::getSelectedData(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters, TO const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareSelectedCreaturesForConversionToTO, includeClusters, data);
    KERNEL_CALL(cudaGetSelectedGenomeData, data, includeClusters, to);
    KERNEL_CALL(cudaGetSelectedCreatureData, data, includeClusters, to);
    KERNEL_CALL(cudaGetSelectedCellDataWithoutConnections, data, includeClusters, to);
    KERNEL_CALL(cudaResolveConnections, data, to);
    KERNEL_CALL(cudaGetSelectedParticleData, data, to);
}

void _DataAccessKernelsService::getInspectedData(CudaSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, TO const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareCreaturesAndGenomesForConversionToTO, entityIds, data);
    KERNEL_CALL(cudaGetGenomeData, entityIds, data, to);
    KERNEL_CALL(cudaGetCreatureData, entityIds, data, to);
    KERNEL_CALL(cudaGetInspectedCellDataWithoutConnections, entityIds, data, to);
    KERNEL_CALL(cudaResolveConnections, data, to);
    KERNEL_CALL(cudaGetInspectedParticleData, entityIds, data, to);
}

void _DataAccessKernelsService::getOverlayData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    TO const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaGetOverlayData, rectUpperLeft, rectLowerRight, data, to);
}

bool _DataAccessKernelsService::getGenomeOfCreature(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t creatureId, TO const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareCreatureGenomeForConversionToTO, creatureId, data);
    setValueToDevice(_foundResult, false);
    KERNEL_CALL(cudaGetGenomeOfCreature, creatureId, data, to, _foundResult);
    cudaDeviceSynchronize();

    return copyToHost(_foundResult);
}

ArraySizesForGpu _DataAccessKernelsService::estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, TO const& to)
{
    setValueToDevice(_arraySizesGPU, ArraySizesForGpu{});
    KERNEL_CALL(cudaEstimateCapacityNeededForGpu, to, _arraySizesGPU);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesGPU);
}

void _DataAccessKernelsService::addData(CudaSettings const& gpuSettings, SimulationData const& data, TO const& to, bool selectData)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);
    KERNEL_CALL(cudaAdaptNumberGenerator, data.primaryNumberGen, to);

    KERNEL_CALL_1_1(cudaGetArraysBasedOnTO, data, to, _cudaCellArray);
    KERNEL_CALL(cudaSetGenomeDataFromTO, data, to);
    KERNEL_CALL(cudaSetCreatureDataFromTO, data, to);
    KERNEL_CALL(cudaSetCellAndParticleDataFromTO, data, to, _cudaCellArray, selectData);
    _garbageCollectorKernels->cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        _editKernels->rolloutSelection(gpuSettings, data);
    }
}

void _DataAccessKernelsService::clearData(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaClearData, data);
}
