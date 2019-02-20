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

	//auxiliary data
	int locked;	//0 = unlocked, 1 = locked
	bool alive;
};

struct ClusterData;
struct CellData
{
	uint64_t id;
	ClusterData* cluster;
	float2 relPos;
	float2 absPos;
	float2 vel;
	float energy;
	int maxConnections;
	int numConnections;
	CellData* connections[MAX_CELL_BONDS];
	CellData* nextTimestep;

	//auxiliary data
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

	//auxiliary data
	bool decompositionRequired;
	int locked;	//0 = unlocked, 1 = locked
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

