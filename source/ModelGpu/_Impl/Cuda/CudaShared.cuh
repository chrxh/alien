#pragma once

#include <cuda_runtime.h>

#include "CudaConstants.cuh"

struct CudaEnergyParticle
{
	float energy;
	float2 pos;
	float2 vel;
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

struct CudaDataForAccess
{
	int numCells;
	CudaCell* cells;
	int numParticles;
	CudaEnergyParticle* particles;
};

extern void cudaInit(int2 const &size);
extern void cudaCalcNextTimestep();
extern CudaDataForAccess cudaGetData();
extern void cudaShutdown();


