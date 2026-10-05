#include "DataAccessKernelsService.cuh"

#include <ranges>

#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/DebugKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>

#include "EditKernelsService.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "SelectionKernelsService.cuh"

void DataAccessKernelsService::init()
{
    getDeviceMemory();
}

void DataAccessKernelsService::shutdown()
{
    for (auto& deviceMemory : _deviceMemories | std::views::values) {
        CudaMemoryManager::getInstance().freeMemory(deviceMemory.cudaCellArray);
        CudaMemoryManager::getInstance().freeMemory(deviceMemory.arraySizesGPU);
        CudaMemoryManager::getInstance().freeMemory(deviceMemory.arraySizesTO);
    }
    _deviceMemories.clear();
}

ArraySizesForTOs DataAccessKernelsService::estimateCapacityNeededForTO(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    auto arraySizesTO = getDeviceMemory().arraySizesTO;
    setValueToDevice(arraySizesTO, ArraySizesForTOs{});
    launchKernelOnDefaultStream(KERNEL(cudaEstimateCapacityNeededForTO_step1), LaunchConfig{launchSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaEstimateCapacityNeededForTO_step2), LaunchConfig{launchSettings.numBlocks, 8}, data, arraySizesTO);
    cudaDeviceSynchronize();

    return copyToHost(arraySizesTO);
}

void DataAccessKernelsService::getData(
    KernelLaunchSettings const& launchSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(
        KERNEL(cudaPrepareCreaturesAndGenomesForConversionToTO), LaunchConfig{launchSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data);
    launchKernelOnDefaultStream(KERNEL(cudaGetGenomeData), LaunchConfig{launchSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetCreatureData), LaunchConfig{launchSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetObjectDataWithoutConnections), LaunchConfig{launchSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaResolveConnections), LaunchConfig{launchSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetParticleData), LaunchConfig{launchSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
}

void DataAccessKernelsService::getSelectedData(KernelLaunchSettings const& launchSettings, SimulationData const& data, bool includeClusters, TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(KERNEL(cudaPrepareSelectedCreaturesForConversionToTO), LaunchConfig{launchSettings.numBlocks, 8}, includeClusters, data);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedGenomeData), LaunchConfig{launchSettings.numBlocks, 8}, data, includeClusters, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedCreatureData), LaunchConfig{launchSettings.numBlocks, 8}, data, includeClusters, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedObjectDataWithoutConnections), LaunchConfig{launchSettings.numBlocks, 8}, data, includeClusters, to);
    launchKernelOnDefaultStream(KERNEL(cudaResolveConnections), LaunchConfig{launchSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedEnergyData), LaunchConfig{launchSettings.numBlocks, 8}, data, to);
}

void DataAccessKernelsService::getInspectedData(
    KernelLaunchSettings const& launchSettings,
    SimulationData const& data,
    InspectedEntityIds entityIds,
    TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(KERNEL(cudaPrepareInspectedCreaturesAndGenomesForConversionToTO), LaunchConfig{launchSettings.numBlocks, 8}, entityIds, data);
    launchKernelOnDefaultStream(KERNEL(cudaGetInspectedGenomeData), LaunchConfig{launchSettings.numBlocks, 8}, entityIds, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetInspectedCreatureData), LaunchConfig{launchSettings.numBlocks, 8}, entityIds, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetInspectedObjectDataWithoutConnections), LaunchConfig{launchSettings.numBlocks, 8}, entityIds, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaResolveConnections), LaunchConfig{launchSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetInspectedEnergyData), LaunchConfig{launchSettings.numBlocks, 8}, entityIds, data, to);
}

void DataAccessKernelsService::getOverlayData(
    KernelLaunchSettings const& launchSettings,
    SimulationData const& data,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetOverlayData), LaunchConfig{launchSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
}

ArraySizesForGpuEntities DataAccessKernelsService::estimateCapacityNeededForGpu(KernelLaunchSettings const& launchSettings, TOs const& to)
{
    auto arraySizesGPU = getDeviceMemory().arraySizesGPU;
    setValueToDevice(arraySizesGPU, ArraySizesForGpuEntities{});
    launchKernelOnDefaultStream(KERNEL(cudaEstimateCapacityNeededForGpu), LaunchConfig{launchSettings.numBlocks, 8}, to, arraySizesGPU);
    cudaDeviceSynchronize();

    return copyToHost(arraySizesGPU);
}

void DataAccessKernelsService::addData(KernelLaunchSettings const& launchSettings, SimulationData const& data, TOs const& to, bool selectData)
{
    auto cudaCellArray = getDeviceMemory().cudaCellArray;
    launchKernelOnDefaultStream(KERNEL(cudaSaveNumEntries), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(KERNEL(cudaAdaptNumberGenerator), LaunchConfig{launchSettings.numBlocks, 8}, data.primaryNumberGen, to);

    launchKernelOnDefaultStream(KERNEL(cudaGetArraysBasedOnTO), LaunchConfig{1, 1}, data, to, cudaCellArray);
    launchKernelOnDefaultStream(KERNEL(cudaSetGenomeDataFromTO), LaunchConfig{launchSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaSetCreatureDataFromTO), LaunchConfig{launchSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaSetCellAndParticleDataFromTO), LaunchConfig{launchSettings.numBlocks, 8}, data, to, cudaCellArray, selectData);
    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(launchSettings, data);
    if (selectData) {
        SelectionKernelsService::get().rolloutSelection(launchSettings, data);
    }
}

void DataAccessKernelsService::clearData(KernelLaunchSettings const& launchSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearData), LaunchConfig{launchSettings.numBlocks, 8}, data);
}

DataAccessKernelsService::DeviceMemory& DataAccessKernelsService::getDeviceMemory()
{
    auto& result = _deviceMemories[getCurrentDevice()];
    if (!result.cudaCellArray) {
        CudaMemoryManager::getInstance().acquireMemory(1, result.cudaCellArray);
        CudaMemoryManager::getInstance().acquireMemory(1, result.arraySizesGPU);
        CudaMemoryManager::getInstance().acquireMemory(1, result.arraySizesTO);
    }
    return result;
}
