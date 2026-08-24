#include "DataAccessKernelsService.cuh"

#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/DebugKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>

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

ArraySizesForTOs DataAccessKernelsService::estimateCapacityNeededForTO(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_arraySizesTO, ArraySizesForTOs{});
    launchKernelOnDefaultStream(KERNEL(cudaEstimateCapacityNeededForTO_step1), LaunchConfig{gpuSettings.numBlocks, 8}, data);
    launchKernelOnDefaultStream(KERNEL(cudaEstimateCapacityNeededForTO_step2), LaunchConfig{gpuSettings.numBlocks, 8}, data, _arraySizesTO);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesTO);
}

void DataAccessKernelsService::getData(
    KernelLaunchSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(
        "cudaPrepareCreaturesAndGenomesForConversionToTO",
        static_cast<void (*)(int2, int2, SimulationData)>(cudaPrepareCreaturesAndGenomesForConversionToTO),
        LaunchConfig{gpuSettings.numBlocks, 8},
        rectUpperLeft,
        rectLowerRight,
        data);
    launchKernelOnDefaultStream(
        "cudaGetGenomeData",
        static_cast<void (*)(int2, int2, SimulationData, TOs)>(cudaGetGenomeData),
        LaunchConfig{gpuSettings.numBlocks, 8},
        rectUpperLeft,
        rectLowerRight,
        data,
        to);
    launchKernelOnDefaultStream(
        "cudaGetCreatureData",
        static_cast<void (*)(int2, int2, SimulationData, TOs)>(cudaGetCreatureData),
        LaunchConfig{gpuSettings.numBlocks, 8},
        rectUpperLeft,
        rectLowerRight,
        data,
        to);
    launchKernelOnDefaultStream(KERNEL(cudaGetObjectDataWithoutConnections), LaunchConfig{gpuSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaResolveConnections), LaunchConfig{gpuSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetParticleData), LaunchConfig{gpuSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
}

void DataAccessKernelsService::getSelectedData(KernelLaunchSettings const& gpuSettings, SimulationData const& data, bool includeClusters, TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(KERNEL(cudaPrepareSelectedCreaturesForConversionToTO), LaunchConfig{gpuSettings.numBlocks, 8}, includeClusters, data);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedGenomeData), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedCreatureData), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedObjectDataWithoutConnections), LaunchConfig{gpuSettings.numBlocks, 8}, data, includeClusters, to);
    launchKernelOnDefaultStream(KERNEL(cudaResolveConnections), LaunchConfig{gpuSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetSelectedEnergyData), LaunchConfig{gpuSettings.numBlocks, 8}, data, to);
}

void DataAccessKernelsService::getInspectedData(
    KernelLaunchSettings const& gpuSettings,
    SimulationData const& data,
    InspectedEntityIds entityIds,
    TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(
        "cudaPrepareCreaturesAndGenomesForConversionToTO",
        static_cast<void (*)(InspectedEntityIds, SimulationData)>(cudaPrepareCreaturesAndGenomesForConversionToTO),
        LaunchConfig{gpuSettings.numBlocks, 8},
        entityIds,
        data);
    launchKernelOnDefaultStream(
        "cudaGetGenomeData",
        static_cast<void (*)(InspectedEntityIds, SimulationData, TOs)>(cudaGetGenomeData),
        LaunchConfig{gpuSettings.numBlocks, 8},
        entityIds,
        data,
        to);
    launchKernelOnDefaultStream(
        "cudaGetCreatureData",
        static_cast<void (*)(InspectedEntityIds, SimulationData, TOs)>(cudaGetCreatureData),
        LaunchConfig{gpuSettings.numBlocks, 8},
        entityIds,
        data,
        to);
    launchKernelOnDefaultStream(KERNEL(cudaGetInspectedObjectDataWithoutConnections), LaunchConfig{gpuSettings.numBlocks, 8}, entityIds, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaResolveConnections), LaunchConfig{gpuSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetInspectedEnergyData), LaunchConfig{gpuSettings.numBlocks, 8}, entityIds, data, to);
}

void DataAccessKernelsService::getOverlayData(
    KernelLaunchSettings const& gpuSettings,
    SimulationData const& data,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    TOs const& to)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearDataTO), LaunchConfig{1, 1}, to);
    launchKernelOnDefaultStream(KERNEL(cudaGetOverlayData), LaunchConfig{gpuSettings.numBlocks, 8}, rectUpperLeft, rectLowerRight, data, to);
}

ArraySizesForGpuEntities DataAccessKernelsService::estimateCapacityNeededForGpu(KernelLaunchSettings const& gpuSettings, TOs const& to)
{
    setValueToDevice(_arraySizesGPU, ArraySizesForGpuEntities{});
    launchKernelOnDefaultStream(KERNEL(cudaEstimateCapacityNeededForGpu), LaunchConfig{gpuSettings.numBlocks, 8}, to, _arraySizesGPU);
    cudaDeviceSynchronize();

    return copyToHost(_arraySizesGPU);
}

void DataAccessKernelsService::addData(KernelLaunchSettings const& gpuSettings, SimulationData const& data, TOs const& to, bool selectData)
{
    launchKernelOnDefaultStream(KERNEL(cudaSaveNumEntries), LaunchConfig{1, 1}, data);
    launchKernelOnDefaultStream(KERNEL(cudaAdaptNumberGenerator), LaunchConfig{gpuSettings.numBlocks, 8}, data.primaryNumberGen, to);

    launchKernelOnDefaultStream(KERNEL(cudaGetArraysBasedOnTO), LaunchConfig{1, 1}, data, to, _cudaCellArray);
    launchKernelOnDefaultStream(KERNEL(cudaSetGenomeDataFromTO), LaunchConfig{gpuSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaSetCreatureDataFromTO), LaunchConfig{gpuSettings.numBlocks, 8}, data, to);
    launchKernelOnDefaultStream(KERNEL(cudaSetCellAndParticleDataFromTO), LaunchConfig{gpuSettings.numBlocks, 8}, data, to, _cudaCellArray, selectData);
    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        SelectionKernelsService::get().rolloutSelection(gpuSettings, data);
    }
}

void DataAccessKernelsService::clearData(KernelLaunchSettings const& gpuSettings, SimulationData const& data)
{
    launchKernelOnDefaultStream(KERNEL(cudaClearData), LaunchConfig{gpuSettings.numBlocks, 8}, data);
}
