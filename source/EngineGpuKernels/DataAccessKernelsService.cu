#include "DataAccessKernelsService.cuh"

#include "DataAccessKernels.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "EditKernelsService.cuh"
#include "DebugKernels.cuh"

_DataAccessKernelsService::_DataAccessKernelsService()
{
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsService>();
    _editKernels = std::make_shared<_EditKernelsService>();

    CudaMemoryManager::getInstance().acquireMemory<Cell*>(1, _cudaCellArray);
    CudaMemoryManager::getInstance().acquireMemory<ObjectTOArraySizes>(1, _arraySizes);
}

_DataAccessKernelsService::~_DataAccessKernelsService()
{
    CudaMemoryManager::getInstance().freeMemory(_cudaCellArray);
    CudaMemoryManager::getInstance().freeMemory(_arraySizes);
}

ObjectTOArraySizes _DataAccessKernelsService::getActualArraySizes(GpuSettings const& gpuSettings, SimulationData const& data)
{
    setValueToDevice(_arraySizes, ObjectTOArraySizes{});
    KERNEL_CALL(cudaGetActualArraySizes, data, _arraySizes);
    return copyToHost(_arraySizes);
}

void _DataAccessKernelsService::getData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetCellDataWithoutConnections, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, data, dataTO);
}

void _DataAccessKernelsService::getSelectedData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    bool includeClusters,
    DataTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetSelectedCellDataWithoutConnections, data, includeClusters, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetSelectedParticleData, data, dataTO);
}

void _DataAccessKernelsService::getInspectedData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    InspectedEntityIds entityIds,
    DataTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetInspectedCellDataWithoutConnections, entityIds, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetInspectedParticleData, entityIds, data, dataTO);
}

void _DataAccessKernelsService::getOverlayData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    DataTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetOverlayData, rectUpperLeft, rectLowerRight, data, dataTO);
}

void _DataAccessKernelsService::addData(GpuSettings const& gpuSettings, SimulationData const& data, DataTO const& dataTO, bool selectData, bool createIds)
{
    KERNEL_CALL_1_1(cudaSaveNumEntries, data);
    KERNEL_CALL(cudaAdaptNumberGenerator, data.numberGen1, dataTO);

    KERNEL_CALL_1_1(cudaGetArraysBasedOnTO, data, dataTO, _cudaCellArray);
    KERNEL_CALL(cudaCreateDataFromTO, data, dataTO, _cudaCellArray, selectData, createIds);
    _garbageCollectorKernels->cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        _editKernels->rolloutSelection(gpuSettings, data);
    }
    KERNEL_CALL(cudaAdaptNumberGenerator, data.numberGen1, dataTO);
}

void _DataAccessKernelsService::clearData(GpuSettings const& gpuSettings, SimulationData const& data)
{
    KERNEL_CALL(cudaClearData, data);
}
