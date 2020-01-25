#pragma once

#include <cuda_runtime.h>

#include "CudaSimulation.cuh"

#define MAX_TOKEN_MEM_SIZE 256
#define MAX_CELL_BONDS 6
#define MAX_CELL_STATIC_BYTES 48
#define MAX_CELL_MUTABLE_BYTES 16

struct TokenAccessTO
{
	float energy;
	char memory[MAX_TOKEN_MEM_SIZE];
	int cellIndex;
};

struct ParticleMetadataAccessTO
{
    unsigned char color;
};

struct ParticleAccessTO
{
	uint64_t id;
	float energy;
	float2 pos;
	float2 vel;
    ParticleMetadataAccessTO metadata;
};

struct CellMetadataAccessTO
{
    unsigned char color;

    int nameLen;
    int nameStringIndex;

    int descriptionLen;
    int descriptionStringIndex;

    int sourceCodeLen;
    int sourceCodeStringIndex;
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
    unsigned char numStaticBytes;
    char staticData[MAX_CELL_STATIC_BYTES];
    unsigned char numMutableBytes;
    char mutableData[MAX_CELL_MUTABLE_BYTES];
    int tokenUsages;
    CellMetadataAccessTO metadata;
};

struct ClusterMetadataAccessTO
{
    int nameLen;
    int nameStringIndex;
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
    ClusterMetadataAccessTO metadata;
};

struct DataAccessTO
{
	int* numClusters = nullptr;
	ClusterAccessTO* clusters = nullptr;
	int* numCells = nullptr;
	CellAccessTO* cells = nullptr;
	int* numParticles = nullptr;
	ParticleAccessTO* particles = nullptr;
	int* numTokens = nullptr;
	TokenAccessTO* tokens = nullptr;
    int* numStringBytes = nullptr;
    char* stringBytes = nullptr;

	bool operator==(DataAccessTO const& other) const
	{
		return numClusters == other.numClusters
			&& clusters == other.clusters
			&& numCells == other.numCells
			&& cells == other.cells
			&& numParticles == other.numParticles
			&& particles == other.particles
			&& numTokens == other.numTokens
			&& tokens == other.tokens
            && numStringBytes == other.numStringBytes
            && stringBytes == other.stringBytes;
	}
};

