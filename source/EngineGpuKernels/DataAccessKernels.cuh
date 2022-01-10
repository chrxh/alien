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
__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, DataAccessTO dataTO);
__global__ void cudaGetSelectedParticleData(SimulationData data, DataAccessTO access);
__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, DataAccessTO dataTO);
__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, DataAccessTO access);
__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO);
__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO dataTO);
__global__ void cudaResolveConnections(SimulationData data, DataAccessTO dataTO);
__global__ void cudaGetTokenData(SimulationData data, DataAccessTO dataTO);
__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataAccessTO access);
__global__ void cudaCreateDataFromTO(SimulationData data, DataAccessTO dataTO, bool selectNewData);
__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, DataAccessTO dataTO);
__global__ void cudaClearDataTO(DataAccessTO dataTO);
__global__ void cudaSaveNumEntries(SimulationData data);
__global__ void cudaGetInspectedSimulationData(SimulationData data, InspectedEntityIds entityIds, DataAccessTO dataTO);
__global__ void cudaClearData(SimulationData data);
