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

    void init(int2 const& universeSize, int newTimestep)
    {
        size = universeSize;
        timestep = newTimestep;

        entities.init();
        entitiesForCleanup.init();
        cellFunctionData.init(universeSize);
        cellMap.init(size);
        particleMap.init(size);

        dynamicMemory.init();
        numberGen.init(40312357);

        numPixels = size.x * size.y;
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(numPixels*2, imageData);
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(1, numOperations);
    }

    void resizeImage(int2 const& newSize)
    {
        CudaMemoryManager::getInstance().freeMemory(numPixels * 2, imageData);
        CudaMemoryManager::getInstance().acquireMemory<unsigned int>(
            max(newSize.x * newSize.y, size.x * size.y)*2, imageData);
        numPixels = newSize.x * newSize.y;
    }

    bool shouldResize(int additionalCells, int additionalParticles, int additionalTokens)
    {
        auto cellAndParticleArraySizeInc = std::max(additionalCells, additionalParticles);
        auto tokenArraySizeInc = std::max(additionalTokens, cellAndParticleArraySizeInc / 10);

        return entities.cells.shouldResize_host(cellAndParticleArraySizeInc)
            || entities.cellPointers.shouldResize_host(cellAndParticleArraySizeInc * 10)
            || entities.particles.shouldResize_host(cellAndParticleArraySizeInc)
            || entities.particlePointers.shouldResize_host(cellAndParticleArraySizeInc * 10)
            || entities.tokens.shouldResize_host(tokenArraySizeInc)
            || entities.tokenPointers.shouldResize_host(tokenArraySizeInc * 10);
    }

    __device__ bool shouldResize()
    {
        return entities.cells.shouldResize(0) || entities.cellPointers.shouldResize(0)
            || entities.particles.shouldResize(0) || entities.particlePointers.shouldResize(0)
            || entities.tokens.shouldResize(0) || entities.tokenPointers.shouldResize(0);
    }

    void resizeTarget(int additionalCells, int additionalParticles, int additionalTokens)
    {
        auto cellAndParticleArraySizeInc = std::max(additionalCells, additionalParticles);
        auto tokenArraySizeInc = std::max(additionalTokens, cellAndParticleArraySizeInc / 10);

        resizeTargetIntern(entities.cells, entitiesForCleanup.cells, cellAndParticleArraySizeInc);
        resizeTargetIntern(entities.cellPointers, entitiesForCleanup.cellPointers, cellAndParticleArraySizeInc * 10);
        resizeTargetIntern(entities.particles, entitiesForCleanup.particles, cellAndParticleArraySizeInc);
        resizeTargetIntern(entities.particlePointers, entitiesForCleanup.particlePointers, cellAndParticleArraySizeInc * 10);
        resizeTargetIntern(entities.tokens, entitiesForCleanup.tokens, tokenArraySizeInc);
        resizeTargetIntern(entities.tokenPointers, entitiesForCleanup.tokenPointers, tokenArraySizeInc * 10);
    }

    bool isEmpty()
    {
        return 0 == entities.cells.getNumEntries_host() && 0 == entities.particles.getNumEntries_host()
            && 0 == entities.tokens.getNumEntries_host();
    }

    void resizeSource()
    {
        entities.cells.resize(entitiesForCleanup.cells.getSize_host());
        entities.cellPointers.resize(entitiesForCleanup.cellPointers.getSize_host());
        entities.particles.resize(entitiesForCleanup.particles.getSize_host());
        entities.particlePointers.resize(entitiesForCleanup.particlePointers.getSize_host());
        entities.tokens.resize(entitiesForCleanup.tokens.getSize_host());
        entities.tokenPointers.resize(entitiesForCleanup.tokenPointers.getSize_host());

        auto cellArraySize = entities.cells.getSize_host();
        cellMap.resize(cellArraySize);
        particleMap.resize(cellArraySize);

        int upperBoundDynamicMemory = sizeof(Operation) * (cellArraySize + 1000);
        dynamicMemory.resize(upperBoundDynamicMemory);
    }

    void swap()
    {
        entities.cells.swapContent_host(entitiesForCleanup.cells);
        entities.cellPointers.swapContent_host(entitiesForCleanup.cellPointers);
        entities.particles.swapContent_host(entitiesForCleanup.particles);
        entities.particlePointers.swapContent_host(entitiesForCleanup.particlePointers);
        entities.tokens.swapContent_host(entitiesForCleanup.tokens);
        entities.tokenPointers.swapContent_host(entitiesForCleanup.tokenPointers);
    }

    void free()
    {
        entities.free();
        entitiesForCleanup.free();
        cellFunctionData.free();
        cellMap.free();
        particleMap.free();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());
        numberGen.free();
        CHECK_FOR_CUDA_ERROR(cudaGetLastError());
        dynamicMemory.free();

        CudaMemoryManager::getInstance().freeMemory(numPixels * 2, imageData);
        CudaMemoryManager::getInstance().freeMemory(1, numOperations);
    }

private:
    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities)
    {
        if (sourceArray.shouldResize_host(additionalEntities)) {
            auto newSize = (sourceArray.getNumEntries_host() + additionalEntities) * 2;
            targetArray.resize(newSize);
        }
    }
};

