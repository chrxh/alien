#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/ZoomLevels.h"

#include "TO.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"
#include "ObjectFactory.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"
#include "CudaGeometryBuffers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void cudaBackground(uint64_t* imageData, int2 imageSize, int2 worldSize, float zoom, float2 rectUpperLeft, float2 rectLowerRight);
__global__ void cudaPrepareFilteringForRendering(Array<Cell*> filteredCells, Array<Particle*> filteredParticles);
__global__ void cudaFilterCellsForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Cell*> cells,
    Array<Cell*> filteredCells,
    int2 imageSize,
    float zoom);
__global__ void cudaFilterParticlesForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Particle*> particles,
    Array<Particle*> filteredParticles,
    int2 imageSize,
    float zoom);
__global__ void cudaDrawCells(
    uint64_t timestep,
    int2 worldSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Cell*> cells,
    uint64_t* imageData,
    int2 imageSize,
    float zoom);
__global__ void cudaDrawCellGlow(int2 worldSize, float2 rectUpperLeft, Array<Cell*> cells, uint64_t* imageData, int2 imageSize, float zoom);

__global__ void cudaDrawParticles(int2 worldSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Particle*> particles, uint64_t* imageData, int2 imageSize, float zoom);
__global__ void cudaDrawRadiationSources(uint64_t* targetImage, float2 rectUpperLeft, int2 worldSize, int2 imageSize, float zoom);
__global__ void cudaDrawRepetition(int2 worldSize, int2 imageSize, float2 rectUpperLeft, float2 rectLowerRight, uint64_t* imageData, float zoom);

__global__ void cudaExtractCellData(SimulationData data, CellVertexData* objectData);
__global__ void cudaExtractEnergyParticleData(SimulationData data, EnergyParticleVertexData* energyParticleData);
__global__ void cudaExtractLocationData(SimulationData data, LocationVertexData* locationData);
__global__ void cudaExtractSelectedObjectData(SimulationData data, SelectedObjectVertexData* selectedObjectData, uint64_t* numSelectedObjects);
__global__ void cudaExtractLineIndices(SimulationData data, unsigned int* lineIndices, uint64_t* numLineIndices);
__global__ void cudaExtractTriangleIndices(SimulationData data, unsigned int* triangleIndices, uint64_t* numTriangleIndices);
__global__ void cudaExtractSelectedConnectionData(SimulationData data, ConnectionArrowVertexData* connectionArrowData, uint64_t* numConnectionArrowVertices);
