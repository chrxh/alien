#pragma once

struct SimulationData;
struct CellAccessTO;
struct ClusterAccessTO;
struct SimulationAccessTO;

class CudaSimulation
{
public:
	CudaSimulation(int2 const &size);
	~CudaSimulation();

	void calcNextTimestep();
	SimulationAccessTO* getAccessData(int2 const& rectUpperLeft, int2 const& rectLowerRight);
	void updateToSimulation();

private:
	void prepareTargetData();
	void swapData();
/*
	void correctPointersAfterCellCopy(CellAccessTO* cell, int64_t addressShiftCell, int64_t addressShiftCluster);
	void correctPointersAfterClusterCopy(ClusterAccessTO* cluster, int64_t addressShiftCell);
*/
	void setCudaSimulationParameters();

	cudaStream_t _cudaStream;
	SimulationData* _data = nullptr;
	int2 _rectUpperLeft = { 0,0 };
	int2 _rectLowerRight = { 0,0 };
	SimulationAccessTO* _accessTO;
	SimulationAccessTO* _cudaAccessTO;
};
