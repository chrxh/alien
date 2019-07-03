#pragma once

struct SimulationData;
struct CellAccessTO;
struct ClusterAccessTO;
struct DataAccessTO;
struct SimulationParameters;
struct CudaConstants;

class CudaSimulation
{
public:
    CudaSimulation(int2 const &size, SimulationParameters const& parameters, CudaConstants const& cudaConstants);
    ~CudaSimulation();

    void calcNextTimestep();

    void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    void setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);

    void setSimulationParameters(SimulationParameters const& parameters);

private:
    void setCudaConstants(CudaConstants const& cudaConstants);
    void DEBUG_printNumEntries();

private:
    SimulationData* _internalData;
    DataAccessTO* _cudaAccessTO;
};
