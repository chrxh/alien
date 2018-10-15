#pragma once

#include "Base.cuh"
#include "CudaInterface.cuh"
#include "Map.cuh"

struct SimulationData
{
	int2 size;

	CellData **cellMap1;
	CellData **cellMap2;

	ParticleData **particleMap1;
	ParticleData **particleMap2;

	ArrayController<ClusterData> clustersAC1;
	ArrayController<ClusterData> clustersAC2;
	ArrayController<CellData> cellsAC1;
	ArrayController<CellData> cellsAC2;
	ArrayController<ParticleData> particlesAC1;
	ArrayController<ParticleData> particlesAC2;

	CudaNumberGenerator numberGen;
};

