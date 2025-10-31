#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/InspectedEntityIds.h"

#include "TO.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "EditKernels.cuh"

#include "SimulationData.cuh"

//tags cell with cellTO index and tags cellTO connections with cell index
__global__ void cudaPrepareCreaturesForConversionToTO(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data);
__global__ void cudaPrepareSelectedCreaturesForConversionToTO(bool includeClusters, SimulationData data);
__global__ void cudaPrepareCreaturesForConversionToTO(InspectedEntityIds ids, SimulationData data);

__global__ void cudaGetSelectedParticleData(SimulationData data, TO access);
__global__ void cudaGetInspectedParticleData(InspectedEntityIds ids, SimulationData data, TO access);
__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO collectionTO);

__global__ void cudaGetCreatureData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO collectionTO);
__global__ void cudaGetSelectedCreatureData(SimulationData data, bool includeClusters, TO collectionTO);
__global__ void cudaGetCreatureData(InspectedEntityIds ids, SimulationData data, TO collectionTO);

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO collectionTO);
__global__ void cudaGetGenomeData(InspectedEntityIds ids, SimulationData data, TO collectionTO);

__global__ void cudaGetCellDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO collectionTO);
__global__ void cudaGetSelectedCellDataWithoutConnections(SimulationData data, bool includeClusters, TO collectionTO);
__global__ void cudaGetInspectedCellDataWithoutConnections(InspectedEntityIds ids, SimulationData data, TO collectionTO);

__global__ void cudaResolveConnections(SimulationData data, TO collectionTO);
__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TO access);
__global__ void cudaGetArraysBasedOnTO(SimulationData data, TO collectionTO, Cell** cellArray);
__global__ void cudaSetCreatureDataFromTO(SimulationData data, TO collectionTO);
__global__ void cudaSetGenomeDataFromTO(SimulationData data, TO collectionTO);
__global__ void cudaSetDataFromTO(SimulationData data, TO collectionTO, Cell** cellArray, bool selectNewData);
__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, TO collectionTO);
__global__ void cudaClearDataTO(TO collectionTO);
__global__ void cudaSaveNumEntries(SimulationData data);
__global__ void cudaClearData(SimulationData data);
__global__ void cudaEstimateCapacityNeededForTO(SimulationData data, ArraySizesForTO* arraySizes);
__global__ void cudaEstimateCapacityNeededForGpu(TO collectionTO, ArraySizesForGpu* arraySizes);
