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

    void init(int2 size_, CudaConstants const& cudaConstants)
    {
        size = size_;
        entities.init(cudaConstants);
        entitiesNew.init(cudaConstants);

        CudaMemoryManager::getInstance().acquireMemory<Cell*>(size.x * size.y, cellMap);
        CudaMemoryManager::getInstance().acquireMemory<Particle*>(size.x * size.y, particleMap);

        std::vector<Cell*> hostCellMap(size.x * size.y, 0);
        std::vector<Particle*> hostParticleMap(size.x * size.y, 0);
        checkCudaErrors(cudaMemcpy(cellMap, hostCellMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(particleMap, hostParticleMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
        numberGen.init(31231257);
    }

    void free()
    {
        entities.free();
        entitiesNew.free();
        CudaMemoryManager::getInstance().freeMemory(cellMap);
        CudaMemoryManager::getInstance().freeMemory(particleMap);
        numberGen.free();
    }
};

