#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"

struct SimulationData
{
    int2 size;

    Map<Cell> cellMap;
    Map<Particle> particleMap;

    Entities entities;
    Entities entitiesForCleanup;

    ArrayController arrays;

    CudaNumberGenerator numberGen;

    void init(int2 size_, CudaConstants const& cudaConstants)
    {
        size = size_;

        entities.init(cudaConstants);
        entitiesForCleanup.init(cudaConstants);
        cellMap.init(size, cudaConstants.MAX_CELLPOINTERS);
        particleMap.init(size, cudaConstants.MAX_PARTICLEPOINTERS);
        arrays.init(cudaConstants.DYNAMIC_MEMORY_SIZE);
        numberGen.init(40312357);
    }

    void free()
    {
        entities.free();
        entitiesForCleanup.free();
        cellMap.free();
        particleMap.free();
        numberGen.free();
        arrays.free();
    }
};

