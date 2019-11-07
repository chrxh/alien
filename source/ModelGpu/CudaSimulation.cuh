#pragma once

#include "ModelBasic/MonitorData.h"

#include "Definitions.cuh"

class CudaSimulation
{
public:
    CudaSimulation(int2 const &size, SimulationParameters const& parameters, CudaConstants const& cudaConstants);
    ~CudaSimulation();

    void calcCudaTimestep();

    void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    void setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);

    MonitorData getMonitorData();

    void setSimulationParameters(SimulationParameters const& parameters);

private:
    void setCudaConstants(CudaConstants const& cudaConstants);
    void DEBUG_printNumEntries();

private:
    SimulationData* _cudaSimulationData;
    DataAccessTO* _cudaAccessTO;
    CudaMonitorData* _cudaMonitorData;
};
