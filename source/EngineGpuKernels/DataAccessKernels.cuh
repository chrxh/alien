#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/InspectedEntityIds.h"
#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "EditKernels.cuh"

#include "SimulationData.cuh"

__device__ extern void copyString(int& targetLen, int& targetStringIndex, int sourceLen, char* sourceString, int& numStringBytes, char*& stringBytes);

__device__ extern void createCellTO(Cell* cell, DataAccessTO& dataTO, Cell* cellArrayStart);

__device__ extern void createParticleTO(Particle* particle, DataAccessTO& dataTO);

__global__ extern void tagCells(int2 rectUpperLeft, int2 rectLowerRight, Array<Cell*> cells);

__global__ extern void rolloutTagToCellClusters(Array<Cell*> cells, int* change);

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ extern void getCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO);

__global__ extern void getInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, DataAccessTO dataTO);

__global__ extern void getSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, DataAccessTO dataTO);

__global__ extern void resolveConnections(SimulationData data, DataAccessTO dataTO);

__global__ extern void getOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO);

__global__ extern void getTokenData(SimulationData data, DataAccessTO dataTO);

__global__ extern void getParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access);

__global__ extern void getInspectedParticleData(InspectedEntityIds ids, SimulationData data, DataAccessTO access);

__global__ extern void getSelectedParticleData(SimulationData data, DataAccessTO access);

__global__ extern void createDataFromTO(SimulationData data, DataAccessTO dataTO, bool selectNewData);

__global__ extern void adaptNumberGenerator(CudaNumberGenerator numberGen, DataAccessTO dataTO);

/************************************************************************/
/* Main      															*/
/************************************************************************/
__global__ extern void clearDataTO(DataAccessTO dataTO);

__global__ extern void prepareSetData(SimulationData data);

__global__ extern void cudaGetSelectedSimulationData(SimulationData data, bool includeClusters, DataAccessTO dataTO);

__global__ extern void cudaGetInspectedSimulationData(SimulationData data, InspectedEntityIds entityIds, DataAccessTO dataTO);

__global__ extern void cudaGetSimulationOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access);

__global__ extern void cudaClearData(SimulationData data);
