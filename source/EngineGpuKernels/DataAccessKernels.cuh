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

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ extern void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO);
__global__ extern void cudaResolveConnections(SimulationData data, DataAccessTO dataTO);
__global__ extern void cudaGetTokenData(SimulationData data, DataAccessTO dataTO);
__global__ extern void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access);
__global__ extern void cudaCreateDataFromTO(SimulationData data, DataAccessTO dataTO, bool selectNewData);
__global__ extern void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, DataAccessTO dataTO);
__global__ extern void cudaClearDataTO(DataAccessTO dataTO);
__global__ extern void cudaPrepareSetData(SimulationData data);
__global__ extern void cudaGetSelectedSimulationData(SimulationData data, bool includeClusters, DataAccessTO dataTO);
__global__ extern void cudaGetInspectedSimulationData(SimulationData data, InspectedEntityIds entityIds, DataAccessTO dataTO);
__global__ extern void cudaGetSimulationOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access);
__global__ extern void cudaClearData(SimulationData data);
