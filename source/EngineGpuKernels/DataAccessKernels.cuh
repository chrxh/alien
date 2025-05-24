#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/InspectedEntityIds.h"

#include "ObjectTO.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "EditKernels.cuh"

#include "SimulationData.cuh"

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaPrepareGenomesForConversionToTO(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data);
__global__ void cudaGetSelectedParticleData(SimulationData data, CollectionTO access);
__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, CollectionTO access);
__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO);

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO);
__global__ void cudaGetSelectedGenomeData(SimulationData data, bool includeClusters, CollectionTO collectionTO);

__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO collectionTO);
__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, CollectionTO collectionTO);
__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, CollectionTO collectionTO);

__global__ void cudaResolveConnections(SimulationData data, CollectionTO collectionTO);
__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, CollectionTO access);
__global__ void cudaGetArraysBasedOnTO(SimulationData data, CollectionTO collectionTO, Cell** cellArray);
__global__ void cudaSetGenomeDataFromTO(SimulationData data, CollectionTO collectionTO, bool createIds);
__global__ void cudaSetDataFromTO(SimulationData data, CollectionTO collectionTO, Cell** cellArray, bool selectNewData, bool createIds);
__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, CollectionTO collectionTO);
__global__ void cudaClearDataTO(CollectionTO collectionTO);
__global__ void cudaSaveNumEntries(SimulationData data);
__global__ void cudaClearData(SimulationData data);
__global__ void cudaEstimateCapacityNeededForTO(SimulationData data, ArraySizesForTO* arraySizes);
__global__ void cudaEstimateCapacityNeededForGpu(CollectionTO collectionTO, ArraySizesForGpu* arraySizes);
