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
    Entities entitiesNew;
    CudaNumberGenerator numberGen;

    void init(int2 size_, CudaConstants const& cudaConstants)
    {
        size = size_;
        entities.init(cudaConstants);
        entitiesNew.init(cudaConstants);

        cellMap.init(size);
        particleMap.init(size);

        numberGen.init(31231257);
    }

    void free()
    {
        entities.free();
        entitiesNew.free();
        cellMap.free();
        particleMap.free();
        numberGen.free();
    }
};

