#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "CudaGeometryBuffers.cuh"
#include "SimulationData.cuh"
#include "TOs.cuh"

__global__ void cudaCorrectPositionsForRendering(SimulationData data, float2 visibleTopLeft);
__global__ void cudaExtractObjectData(SimulationData data, ObjectVertexData* objectData, uint64_t* numObjects);
__global__ void cudaExtractFluidParticleData(SimulationData data, FluidParticleVertexData* fluidParticleData, uint64_t* numFluidParticles);
__global__ void cudaExtractLocationData(SimulationData data, LocationVertexData* locationData, uint64_t* numLocations, float2 visibleTopLeft);
__global__ void cudaExtractSelectedObjectData(SimulationData data, SelectedObjectVertexData* selectedObjectData, uint64_t* numSelectedObjects);
__global__ void cudaExtractSelectedConnectionData(SimulationData data, ConnectionArrowVertexData* connectionArrowData, uint64_t* numConnectionArrowVertices);
__global__ void cudaExtractLineIndices(SimulationData data, unsigned int* lineIndices, uint64_t* numLineIndices);
__global__ void cudaExtractTriangleIndices(SimulationData data, unsigned int* triangleIndices, uint64_t* numTriangleIndices);
__global__ void cudaExtractAttackEventData(SimulationData data, AttackEventVertexData* attackEventData, uint64_t* numAttackEventVertices);
__global__ void cudaExtractDetonationEventData(SimulationData data, DetonationEventVertexData* detonationEventData, uint64_t* numDetonationEventVertices);
