#pragma once

#include <EngineInterface/InspectedEntityIds.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "EditKernels.cuh"
#include "GarbageCollectorKernels.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "SimulationData.cuh"
#include "TOs.cuh"

__global__ void cudaPrepareCreaturesAndGenomesForConversionToTO(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data);
__global__ void cudaPrepareSelectedCreaturesForConversionToTO(bool includeClusters, SimulationData data);
__global__ void cudaPrepareCreaturesAndGenomesForConversionToTO(InspectedEntityIds ids, SimulationData data);
__global__ void cudaPrepareCreatureGenomeForConversionToTO(uint64_t creatureId, SimulationData data);

__global__ void cudaGetSelectedEnergyData(SimulationData data, TOs access);
__global__ void cudaGetInspectedEnergyData(InspectedEntityIds ids, SimulationData data, TOs access);
__global__ void cudaGetOverlayData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to);

__global__ void cudaGetGenomeData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to);
__global__ void cudaGetSelectedGenomeData(SimulationData data, bool includeClusters, TOs to);
__global__ void cudaGetGenomeData(InspectedEntityIds ids, SimulationData data, TOs to);

__global__ void cudaGetCreatureData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to);
__global__ void cudaGetSelectedCreatureData(SimulationData data, bool includeClusters, TOs to);
__global__ void cudaGetCreatureData(InspectedEntityIds ids, SimulationData data, TOs to);

__global__ void cudaGetGenomeOfCreature(uint64_t creatureId, SimulationData data, TOs to, bool* found);

__global__ void cudaGetObjectDataWithoutConnections(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs to);
__global__ void cudaGetSelectedObjectDataWithoutConnections(SimulationData data, bool includeClusters, TOs to);
__global__ void cudaGetInspectedObjectDataWithoutConnections(InspectedEntityIds ids, SimulationData data, TOs to);

__global__ void cudaResolveConnections(SimulationData data, TOs to);
__global__ void cudaGetParticleData(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data, TOs access);
__global__ void cudaGetArraysBasedOnTO(SimulationData data, TOs to, Object** cellArray);

__global__ void cudaSetGenomeDataFromTO(SimulationData data, TOs to);
__global__ void cudaSetCreatureDataFromTO(SimulationData data, TOs to);
__global__ void cudaSetCellAndParticleDataFromTO(SimulationData data, TOs to, Object** cellArray, bool selectNewData);

__global__ void cudaAdaptNumberGenerator(CudaNumberGenerator numberGen, TOs to);
__global__ void cudaClearDataTO(TOs to);
__global__ void cudaSaveNumEntries(SimulationData data);
__global__ void cudaClearData(SimulationData data);

__global__ void cudaEstimateCapacityNeededForTO_step1(SimulationData data);
__global__ void cudaEstimateCapacityNeededForTO_step2(SimulationData data, ArraySizesForTOs* arraySizes);
__global__ void cudaEstimateCapacityNeededForGpu(TOs to, ArraySizesForGpuEntities* arraySizes);
