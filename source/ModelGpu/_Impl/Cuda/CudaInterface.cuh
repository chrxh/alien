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

struct CellClusterData;
struct CellData
{
	uint64_t id;
	CellClusterData* cluster;
	float2 relPos;
	float2 absPos;
	float energy;
	int numConnections;
	CellData* connections[CELL_MAX_BONDS];
	CellData* nextTimestep;
	int protectionCounter;
	bool setProtectionCounterForNextTimestep;
};

struct CellClusterData
{
	float2 pos;
	float2 vel;
	float angle;
	float angularVel;
	float angularMass;
	int numCells;
	CellData* cells;
};

struct DataForAccess
{
	int numClusters;
	CellClusterData* clusters;
	int numCells;
	CellData* cells;
	int numParticles;
	ParticleData* particles;
};

extern void cudaInit(int2 const &size);
extern void cudaCalcNextTimestep();
extern DataForAccess cudaGetData();
extern void cudaDataPtrCorrection();
extern void cudaShutdown();


