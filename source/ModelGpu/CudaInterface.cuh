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
	bool setProtectionCounterForNextTimestep;
	bool alive;
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

extern void cudaInit(int2 const &size);
extern void cudaCalcNextTimestep();
extern SimulationDataForAccess cudaGetData();
extern void cudaDataPtrCorrection();
extern void cudaShutdown();


