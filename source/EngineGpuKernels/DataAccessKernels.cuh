#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/InspectedEntityIds.h"
#include "TOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "EditKernels.cuh"

#include "SimulationData.cuh"

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, DataTO dataTO);
__global__ void cudaGetSelectedParticleData(SimulationData data, DataTO access);
__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, DataTO dataTO);
__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, DataTO access);
__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataTO dataTO);
__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataTO dataTO);
__global__ void cudaResolveConnections(SimulationData data, DataTO dataTO);
__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, DataTO access);
__global__ void cudaCreateDataFromTO(SimulationData data, DataTO dataTO, bool selectNewData, bool createIds);
__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, DataTO dataTO);
__global__ void cudaClearDataTO(DataTO dataTO);
__global__ void cudaSaveNumEntries(SimulationData data);
__global__ void cudaClearData(SimulationData data);
