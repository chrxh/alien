#include "DataAccessKernelsLauncher.cuh"

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
void DataAccessKernelsLauncher::getData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetCellDataWithoutConnections, rectUpperLeft, rectLowerRight, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetTokenData, data, dataTO);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, data, dataTO);
}

void DataAccessKernelsLauncher::getSelectedData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    bool includeClusters,
    DataAccessTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetSelectedCellDataWithoutConnections, data, includeClusters, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetTokenData, data, dataTO);
    KERNEL_CALL(cudaGetSelectedParticleData, data, dataTO);
}

void DataAccessKernelsLauncher::getInspectedData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    InspectedEntityIds entityIds,
    DataAccessTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetInspectedCellDataWithoutConnections, entityIds, data, dataTO);
    KERNEL_CALL(cudaResolveConnections, data, dataTO);
    KERNEL_CALL(cudaGetTokenData, data, dataTO);
    KERNEL_CALL(cudaGetInspectedParticleData, entityIds, data, dataTO);
}

void DataAccessKernelsLauncher::getOverlayData(
    GpuSettings const& gpuSettings,
    SimulationData const& data,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    DataAccessTO dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetOverlayData, rectUpperLeft, rectLowerRight, data, dataTO);
}

void DataAccessKernelsLauncher::addData(GpuSettings const& gpuSettings, SimulationData data, DataAccessTO dataTO, bool selectData)
{
    KERNEL_CALL_1_1(cudaPrepareSetData, data);
    KERNEL_CALL(cudaAdaptNumberGenerator, data.numberGen, dataTO);
    KERNEL_CALL(cudaCreateDataFromTO, data, dataTO, selectData);
    _garbageCollector.cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        KERNEL_CALL_1_1(cudaRolloutSelection, data);
    }

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
}

void DataAccessKernelsLauncher::clearData(GpuSettings const& gpuSettings, SimulationData data) {}
