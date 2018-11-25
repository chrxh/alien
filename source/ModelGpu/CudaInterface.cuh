#pragma once

#include <cuda_runtime.h>

#include "Constants.cuh"


struct ParticleData
{
	uint64_t id;
	float energy;
	float2 pos;
	float2 vel;
};

struct ClusterData;
struct CellData
{
	uint64_t id;
	ClusterData* cluster;
	float2 relPos;
	float2 absPos;
	float energy;
	int numConnections;
	CellData* connections[CELL_MAX_BONDS];
	CellData* nextTimestep;

	int protectionCounter;
	bool alive;
	int tag;
};

struct ClusterData
{
	uint64_t id;
	float2 pos;
	float2 vel;
	float angle;
	float angularVel;
	float angularMass;
	int numCells;
	CellData* cells;

	bool decompositionRequired;
};

struct SimulationDataForAccess
{
	int numClusters;
	ClusterData* clusters;
	int numCells;
	CellData* cells;
	int numParticles;
	ParticleData* particles;
};

struct SimulationDataInternal;
class CudaSimulator
{
public:
	cudaStream_t cudaStream;
	SimulationDataInternal* data;
	SimulationDataForAccess access;

	CudaSimulator(int2 const &size);
	~CudaSimulator();

	void calcNextTimestep();
	SimulationDataForAccess getDataForAccess();
	void setDataForAccess(SimulationDataForAccess const& newAccess);

private:
	void prepareTargetData();
	void swapData();
	void correctPointersAfterCellCopy(CellData* cell, int64_t addressShiftCell, int64_t addressShiftCluster);
	void correctPointersAfterClusterCopy(ClusterData* cluster, int64_t addressShiftCell);
};
