#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"

struct SimulationData
{
    int2 size;

    Cell** cellMap;
    Particle** particleMap;

    Entities entities;
    Entities entitiesNew;
    CudaNumberGenerator numberGen;

    void init(int2 size_)
    {
        size = size_;
        entities.init();
        entitiesNew.init();

        checkCudaErrors(cudaMalloc(&cellMap, size.x * size.y * sizeof(Cell*)));
        checkCudaErrors(cudaMalloc(&particleMap, size.x * size.y * sizeof(Particle*)));

        std::vector<Cell*> hostCellMap(size.x * size.y, 0);
        std::vector<Particle*> hostParticleMap(size.x * size.y, 0);
        checkCudaErrors(cudaMemcpy(cellMap, hostCellMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(particleMap, hostParticleMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
        numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);
    }

    void free()
    {
        entities.free();
        entitiesNew.free();
        checkCudaErrors(cudaFree(cellMap));
        checkCudaErrors(cudaFree(particleMap));
        numberGen.free();
    }
};

