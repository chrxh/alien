#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "DataAccessKernels.cuh"
#include "Macros.cuh"

class DataAccessKernelLauncher
{
public:
    void getData(
        GpuSettings const& gpuSettings,
        SimulationData const& simulationData,
        int2 const& rectUpperLeft,
        int2 const& rectLowerRight,
        DataAccessTO const& dataTO);

private:
    bool* _cudaBool;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
void DataAccessKernelLauncher::getData(
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
