#pragma once

#include <cuda_runtime.h>

struct CellCuda
{
	double2 relPos;
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

extern void init_Cuda();
extern void calcNextTimestep_Cuda();
extern void getData_Cuda(int& numClusters, ClusterCuda*& clusters);
extern void end_Cuda();

