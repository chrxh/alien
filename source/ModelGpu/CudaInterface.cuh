#pragma once

#include <cuda_runtime.h>

#include "CudaSimulator.cuh"
#include "CudaConstants.cuh"

struct ParticleData
{
	uint64_t id;
	float energy;
	float2 pos;
	float2 vel;
	int locked;	//0 = unlocked, 1 = locked
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
	CellData* connections[MAX_CELL_BONDS];
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

