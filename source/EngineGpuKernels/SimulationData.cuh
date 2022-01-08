#pragma once

#include <atomic>

#include "Base.cuh"
#include "CellFunctionData.cuh"
#include "Definitions.cuh"
#include "EngineInterface/GpuSettings.h"
#include "Entities.cuh"
#include "Map.cuh"
#include "Operation.cuh"
#include "Token.cuh"

struct SimulationData
{
    int2 size;

    CellMap cellMap;
    ParticleMap particleMap;
    CellFunctionData cellFunctionData;

    Entities entities;
    Entities entitiesForCleanup;

    ArraySizes* originalArraySizes;

    unsigned int* numOperations;
    Operation** operations;  //uses dynamic memory

    DynamicMemory dynamicMemory;
    CudaNumberGenerator numberGen;

    void init(int2 const& universeSize);
    bool shouldResize(int additionalCells, int additionalParticles, int additionalTokens);
    void resizeEntitiesForCleanup(int additionalCells, int additionalParticles, int additionalTokens);
    void resizeRemainings();
    bool isEmpty();
    void swap();
    void free();

    __device__ void prepareForNextTimestep();
    __device__ int getMaxOperations() { return entities.cellPointers.getNumEntries(); }
    __device__ bool shouldResize();

private:
    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities);
};
