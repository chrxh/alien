#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_STRING_BYTES 50000000
#define MAX_TOKEN_MEM_SIZE 256
#define MAX_CELL_BONDS 6
#define MAX_CELL_STATIC_BYTES 48
#define MAX_CELL_MUTABLE_BYTES 16

struct TokenAccessTO
{
	float energy;
	char memory[MAX_TOKEN_MEM_SIZE];
	int cellIndex;

	//only for temporary use
	int sequenceNumber;
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

	int selected;
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

struct ConnectionAccessTO
{
    int cellIndex;
    float distance;
    float angleFromPrevious;
};

struct CellAccessTO
{
	uint64_t id;
	float2 pos;
    float2 vel;
	float energy;
	int maxConnections;
	int numConnections;
	int branchNumber;
	bool tokenBlocked;
    ConnectionAccessTO connections[MAX_CELL_BONDS];
    int cellFunctionType;
    unsigned char numStaticBytes;
    char staticData[MAX_CELL_STATIC_BYTES];
    unsigned char numMutableBytes;
    char mutableData[MAX_CELL_MUTABLE_BYTES];
    int cellFunctionInvocations;
    CellMetadataAccessTO metadata;
	bool barrier;

	int selected;
};

struct DataAccessTO
{
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
		return numCells == other.numCells
			&& cells == other.cells
			&& numParticles == other.numParticles
			&& particles == other.particles
			&& numTokens == other.numTokens
			&& tokens == other.tokens
            && numStringBytes == other.numStringBytes
            && stringBytes == other.stringBytes;
	}
};

