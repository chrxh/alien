#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.cuh"
#include "CudaSimulation.cuh"

struct TokenAccessTO
{
	float energy;
	char memory[MAX_TOKEN_MEM_SIZE];
	int cellIndex;
};

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
	int branchNumber;
	bool tokenBlocked;
	int connectionIndices[MAX_CELL_BONDS];
    int cellFunctionType;
    char staticData[MAX_CELL_STATIC_BYTES];
    char mutableData[MAX_CELL_MUTABLE_BYTES];
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
	int numTokens;
	int tokenStartIndex;
};

struct DataAccessTO
{
	int* numClusters;
	ClusterAccessTO* clusters;
	int* numCells;
	CellAccessTO* cells;
	int* numParticles;
	ParticleAccessTO* particles;
	int* numTokens;
	TokenAccessTO* tokens;

	bool operator==(DataAccessTO const& other) const
	{
		return numClusters == other.numClusters
			&& clusters == other.clusters
			&& numCells == other.numCells
			&& cells == other.cells
			&& numParticles == other.numParticles
			&& particles == other.particles
			&& numTokens == other.numTokens
			&& tokens == other.tokens;
	}
};

