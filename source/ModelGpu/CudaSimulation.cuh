#pragma once

struct SimulationData;
struct CellAccessTO;
struct ClusterAccessTO;
struct DataAccessTO;

class CudaSimulation
{
public:
	CudaSimulation(int2 const &size);
	~CudaSimulation();

	void calcNextTimestep();

	void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
	void setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);

private:
	void prepareTargetData();
	void swapData();
	void setCudaSimulationParameters();

	cudaStream_t _cudaStream;
	SimulationData* _internalData;
	DataAccessTO* _cudaAccessTO;
};
