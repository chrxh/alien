#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.cuh"


struct ClusterCuda;
struct CellCuda
{
	ClusterCuda* cluster;
	float2 relPos;
	float2 absPos;
	int numConnections;
	CellCuda* connections[CELL_MAX_BONDS];
	CellCuda* nextTimestep;
	int protectionCounter;
};

struct ClusterCuda
{
	float2 pos;
	float2 vel;
	float angle;
	float angularVel;
	float angularMass;
	int numCells;
	CellCuda* cells;
};

extern void init_Cuda(int2 size);
extern void calcNextTimestep_Cuda();
extern void getDataRef_Cuda(int& numClusters, ClusterCuda*& clusters);
extern void end_Cuda();


