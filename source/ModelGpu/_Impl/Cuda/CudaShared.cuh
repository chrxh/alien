#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.cuh"


struct ClusterCuda;
struct CellCuda
{
	ClusterCuda* cluster;
	double2 relPos;
	double2 absPos;
	int numConnections;
	CellCuda* connections[MAX_CELL_CONNECTIONS];

	CellCuda* nextTimestep;
};

struct ClusterCuda
{
	double2 pos;
	double2 vel;
	double angle;
	double angularVel;
	int numCells;
	CellCuda* cells;
};

extern void init_Cuda(int2 size);
extern void calcNextTimestep_Cuda();
extern void getDataRef_Cuda(int& numClusters, ClusterCuda*& clusters);
extern void end_Cuda();