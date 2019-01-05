#pragma once

struct SimulationDataInternal;
struct CellData;
struct ClusterData;
struct SimulationDataForAccess;

class CudaSimulator
{
public:
	CudaSimulator(int2 const &size);
	~CudaSimulator();

	void calcNextTimestep();
	SimulationDataForAccess const& getDataForAccess();
	void setDataForAccess(SimulationDataForAccess const& newAccess);

private:
	void prepareTargetData();
	void swapData();
	void correctPointersAfterCellCopy(CellData* cell, int64_t addressShiftCell, int64_t addressShiftCluster);
	void correctPointersAfterClusterCopy(ClusterData* cluster, int64_t addressShiftCell);
	void setCudaSimulationParameters();

	cudaStream_t _cudaStream;
	SimulationDataInternal* _data = nullptr;
	SimulationDataForAccess* _access = nullptr;
};
