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
	CellCuda* connections[CELL_MAX_BONDS];
	CellCuda* nextTimestep;
};

struct ClusterCuda
{
	double2 pos;
	double2 vel;
	double angle;
	double angularVel;
	double angularMass;
	int numCells;
	CellCuda* cells;

	inline void updateAngularMass();
	inline void updateRelPos();
};

extern void init_Cuda(int2 size);
extern void calcNextTimestep_Cuda();
extern void getDataRef_Cuda(int& numClusters, ClusterCuda*& clusters);
extern void end_Cuda();


//***** implementations *******

void ClusterCuda::updateAngularMass()
{
	angularMass = 0.0;
	for (int i = 0; i < numCells; ++i) {
		auto relPos = cells[i].relPos;
		angularMass += relPos.x*relPos.x + relPos.y*relPos.y;
	}
}

void ClusterCuda::updateRelPos()
{
	double2 center = { 0.0, 0.0 };
	for (int i = 0; i < numCells; ++i) {
		auto const &relPos = cells[i].relPos;
		center.x += relPos.x;
		center.y += relPos.y;
	}
	center.x /= static_cast<double>(numCells);
	center.y /= static_cast<double>(numCells);
	for (int i = 0; i < numCells; ++i) {
		auto &relPos = cells[i].relPos;
		relPos.x -= center.x;
		relPos.y -= center.y;
	}
}
