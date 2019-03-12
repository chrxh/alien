#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.cuh"
#include "CudaSimulation.cuh"

struct ParticleAccessTO
{
	uint64_t id;
	float energy;
	float2 pos;
	float2 vel;
};

struct CellAccessTO
{
	uint64_t id;
	float2 pos;
	float energy;
	int maxConnections;
	int numConnections;
	int connectionIndices[MAX_CELL_BONDS];
};

struct ClusterAccessTO
{
	uint64_t id;
	float2 pos;
	float2 vel;
	float angle;
	float angularVel;
	int numCells;
	int cellStartIndex;
};

struct DataAccessTO
{
	int* numClusters;
	ClusterAccessTO* clusters;
	int* numCells;
	CellAccessTO* cells;
	int* numParticles;
	ParticleAccessTO* particles;
};

