#pragma once

struct SimulationData;
struct CellAccessTO;
struct ClusterAccessTO;
struct DataAccessTO;
struct SimulationParameters;

class CudaSimulation
{
public:
	CudaSimulation(int2 const &size, SimulationParameters const& parameters);
	~CudaSimulation();

	void calcNextTimestep();

	void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
	void setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);

	void setSimulationParameters(SimulationParameters const& parameters);


private:
	void prepareTargetData();
	void swapData();

	cudaStream_t _cudaStream;
	SimulationData* _internalData;
	DataAccessTO* _cudaAccessTO;
};
