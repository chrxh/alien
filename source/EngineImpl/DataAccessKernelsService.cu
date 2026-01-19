#include "DataAccessKernelsService.cuh"

#include <EngineGpuKernels/DataAccessKernels.cuh>
#include <EngineGpuKernels/DebugKernels.cuh>

#include "EditKernelsService.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "SelectionKernelsService.cuh"

void DataAccessKernelsService::init()
{
    CudaMemoryManager::getInstance().acquireMemory(1, _cudaCellArray);
    CudaMemoryManager::getInstance().acquireMemory(1, _arraySizesGPU);
    CudaMemoryManager::getInstance().acquireMemory(1, _arraySizesTO);
    CudaMemoryManager::getInstance().acquireMemory(1, _foundResult);
}

void DataAccessKernelsService::shutdown()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaCellArray);
    CudaMemoryManager::getInstance().freeMemory(_arraySizesGPU);
    CudaMemoryManager::getInstance().freeMemory(_arraySizesTO);
    CudaMemoryManager::getInstance().freeMemory(_foundResult);
}

ArraySizesForTOs DataAccessKernelsService::estimateCapacityNeededForTO(CudaSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_arraySizesTO, ArraySizesForTOs{});
    KERNEL_CALL(cudaEstimateCapacityNeededForTO_step1, data);
    KERNEL_CALL(cudaEstimateCapacityNeededForTO_step2, data, _arraySizesTO);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesTO);
}

void DataAccessKernelsService::getData(
    CudaSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    TOs const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareCreaturesAndGenomesForConversionToTO, rectUpperLeft, rectLowerRight, data);
    KERNEL_CALL(cudaGetGenomeData, rectUpperLeft, rectLowerRight, data, to);
    KERNEL_CALL(cudaGetCreatureData, rectUpperLeft, rectLowerRight, data, to);
    KERNEL_CALL(cudaGetObjectDataWithoutConnections, rectUpperLeft, rectLowerRight, data, to);
    KERNEL_CALL(cudaResolveConnections, data, to);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, data, to);
}

void DataAccessKernelsService::getSelectedData(CudaSettings const& gpuSettings, SimulationData const& data, bool includeClusters, TOs const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareSelectedCreaturesForConversionToTO, includeClusters, data);
    KERNEL_CALL(cudaGetSelectedGenomeData, data, includeClusters, to);
    KERNEL_CALL(cudaGetSelectedCreatureData, data, includeClusters, to);
    KERNEL_CALL(cudaGetSelectedObjectDataWithoutConnections, data, includeClusters, to);
    KERNEL_CALL(cudaResolveConnections, data, to);
    KERNEL_CALL(cudaGetSelectedEnergyData, data, to);
}

void DataAccessKernelsService::getInspectedData(CudaSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, TOs const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareCreaturesAndGenomesForConversionToTO, entityIds, data);
    KERNEL_CALL(cudaGetGenomeData, entityIds, data, to);
    KERNEL_CALL(cudaGetCreatureData, entityIds, data, to);
    KERNEL_CALL(cudaGetInspectedObjectDataWithoutConnections, entityIds, data, to);
    KERNEL_CALL(cudaResolveConnections, data, to);
    KERNEL_CALL(cudaGetInspectedEnergyData, entityIds, data, to);
}

void DataAccessKernelsService::getOverlayData(CudaSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, TOs const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaGetOverlayData, rectUpperLeft, rectLowerRight, data, to);
}

bool DataAccessKernelsService::getGenomeOfCreature(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t creatureId, TOs const& to)
{
    KERNEL_CALL_1_1(cudaClearDataTO, to);
    KERNEL_CALL(cudaPrepareCreatureGenomeForConversionToTO, creatureId, data);
    cudaDeviceSynchronize();
    setValueToDevice(_foundResult, false);
    KERNEL_CALL(cudaGetGenomeOfCreature, creatureId, data, to, _foundResult);
    cudaDeviceSynchronize();

    return copyToHost(_foundResult);
}

ArraySizesForGpuEntities DataAccessKernelsService::estimateCapacityNeededForGpu(CudaSettings const& gpuSettings, TOs const& to)
{
    setValueToDevice(_arraySizesGPU, ArraySizesForGpuEntities{});
    KERNEL_CALL(cudaEstimateCapacityNeededForGpu, to, _arraySizesGPU);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesGPU);
}

void DataAccessKernelsService::addData(CudaSettings const& gpuSettings, SimulationData const& data, TOs const& to, bool selectData)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);
    KERNEL_CALL(cudaAdaptNumberGenerator, data.primaryNumberGen, to);

    KERNEL_CALL_1_1(cudaGetArraysBasedOnTO, data, to, _cudaCellArray);
    KERNEL_CALL(cudaSetGenomeDataFromTO, data, to);
    KERNEL_CALL(cudaSetCreatureDataFromTO, data, to);
    KERNEL_CALL(cudaSetCellAndParticleDataFromTO, data, to, _cudaCellArray, selectData);
    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        SelectionKernelsService::get().rolloutSelection(gpuSettings, data);
    }
}

void DataAccessKernelsService::clearData(CudaSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaClearData, data);
}
