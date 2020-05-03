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

    int* debug;

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

        CudaMemoryManager::getInstance().acquireMemory<int>(1, debug);
        checkCudaErrors(cudaMemset(debug, 0, sizeof(int)));

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
        CudaMemoryManager::getInstance().freeMemory(debug);
    }
};

