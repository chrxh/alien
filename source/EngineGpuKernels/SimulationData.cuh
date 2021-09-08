#pragma once

#include "EngineInterface/GpuConstants.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"
#include "CellFunctionData.cuh"
#include "Operation.cuh"

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

    unsigned int* numOperations;
    Operation* operations;  //uses dynamic memory

    CudaNumberGenerator numberGen;
    int numPixels;
    unsigned int* imageData;

    void init(int2 const& universeSize, GpuConstants const& gpuConstants, int timestep_)
    {
        size = universeSize;
        timestep = timestep_;

        entities.init(gpuConstants);
        entitiesForCleanup.init(gpuConstants);
        cellFunctionData.init(universeSize);
        cellMap.init(size, gpuConstants.MAX_CELLPOINTERS);
        particleMap.init(size, gpuConstants.MAX_PARTICLEPOINTERS);

        int upperBoundDynamicMemory = sizeof(Operation) * gpuConstants.MAX_CELLPOINTERS + 1000;
        dynamicMemory.init(upperBoundDynamicMemory);
        numberGen.init(40312357);

        numPixels = size.x * size.y;
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(numPixels*2, imageData);
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, numOperations);
    }

    void resizeImage(int2 const& newSize)
    {
        CudaMemoryManager::getInstance().freeMemory(imageData);
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(
            max(newSize.x * newSize.y, size.x * size.y)*2, imageData);
        numPixels = newSize.x * newSize.y;
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
        CudaMemoryManager::getInstance().freeMemory(numOperations);
    }
};

