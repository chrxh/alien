#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"
#include "CellFunctionData.cuh"

struct SimulationData
{
    int2 size;
    int timestep;

    CellMap cellMap;
    ParticleMap particleMap;
    CellFunctionData cellFunctionData;

    Entities entities;
    Entities entitiesForCleanup;

    DynamicMemory dynamicMemory;

    CudaNumberGenerator numberGen;
    int numImageBytes;
    unsigned int* imageData;

    void init(int2 const& universeSize, CudaConstants const& cudaConstants, int timestep_)
    {
        size = universeSize;
        timestep = timestep_;

        entities.init(cudaConstants);
        entitiesForCleanup.init(cudaConstants);
        cellFunctionData.init(universeSize);
        cellMap.init(size, cudaConstants.MAX_CELLPOINTERS, entities.cellPointers.getArrayForHost());
        particleMap.init(size, cudaConstants.MAX_PARTICLEPOINTERS);
        dynamicMemory.init(cudaConstants.DYNAMIC_MEMORY_SIZE);
        numberGen.init(40312357);

        numImageBytes = size.x * size.y;
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(numImageBytes, imageData);
    }

    void resizeImage(int2 const& newSize)
    {
        CudaMemoryManager::getInstance().freeMemory(imageData);
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(max(newSize.x * newSize.y, size.x * size.y), imageData);
        numImageBytes = newSize.x * newSize.y;
    }

    void free()
    {
        entities.free();
        entitiesForCleanup.free();
        cellFunctionData.free();
        cellMap.free();
        particleMap.free();
        numberGen.free();
        dynamicMemory.free();

        CudaMemoryManager::getInstance().freeMemory(imageData);
    }
};

