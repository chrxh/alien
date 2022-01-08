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
    KERNEL_CALL_1_1(clearDataTO, dataTO);
    KERNEL_CALL(getCellDataWithoutConnections, rectUpperLeft, rectLowerRight, simulationData, dataTO);
    KERNEL_CALL(resolveConnections, simulationData, dataTO);
    KERNEL_CALL(getTokenData, simulationData, dataTO);
    KERNEL_CALL(getParticleData, rectUpperLeft, rectLowerRight, simulationData, dataTO);

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
}

void DataAccessKernelsLauncher::addData(GpuSettings const& gpuSettings, SimulationData data, DataAccessTO dataTO, bool selectData)
{
    KERNEL_CALL_1_1(prepareSetData, data);
    KERNEL_CALL(adaptNumberGenerator, data.numberGen, dataTO);
    KERNEL_CALL(createDataFromTO, data, dataTO, selectData);
    _garbageCollector.cleanupAfterDataManipulation(gpuSettings, data);
    if (selectData) {
        KERNEL_CALL_1_1(cudaRolloutSelection, data);
    }

    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
}

void DataAccessKernelsLauncher::clearData(GpuSettings const& gpuSettings, SimulationData data) {}
