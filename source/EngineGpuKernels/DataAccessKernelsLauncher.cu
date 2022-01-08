#include "DataAccessKernelsLauncher.cuh"

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
void DataAccessKernelsLauncher::getData(
    GpuSettings const& gpuSettings,
    SimulationData const& simulationData,
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
{
    KERNEL_CALL_1_1(cudaClearDataTO, dataTO);
    KERNEL_CALL(cudaGetCellDataWithoutConnections, rectUpperLeft, rectLowerRight, simulationData, dataTO);
    KERNEL_CALL(cudaResolveConnections, simulationData, dataTO);
    KERNEL_CALL(cudaGetTokenData, simulationData, dataTO);
    KERNEL_CALL(cudaGetParticleData, rectUpperLeft, rectLowerRight, simulationData, dataTO);

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
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
