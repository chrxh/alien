#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.cuh"

struct CudaLightCell
{
	float2 absPos;
};

struct CudaLightCellCluster
{
	CudaLightCell* cells;
};


struct CudaCellCluster;
struct CudaCell
{
	CudaCellCluster* cluster;
	float2 relPos;
	float2 absPos;
	int numConnections;
	CudaCell* connections[CELL_MAX_BONDS];
	CudaCell* nextTimestep;
	int protectionCounter;
};

struct CudaCellCluster
{
	float2 pos;
	float2 vel;
	float angle;
	float angularVel;
	float angularMass;
	int numCells;
	CudaCell* cells;
};

extern void cudaInit(int2 const &size);
extern void cudaCalcNextTimestep();
extern void cudaGetSimulationDataRef(int& numClusters, CudaCellCluster*& clusters);
extern void cudaShutdown();


