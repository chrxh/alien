#pragma once

#include <atomic>

#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Entities.cuh"
#include "CellFunctionData.cuh"
#include "Operation.cuh"
#include "Map.cuh"

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

    __device__ void prepareForNextTimestep();

    __device__ int getMaxOperations();

    bool shouldResize(int additionalCells, int additionalParticles, int additionalTokens);

    __device__ bool shouldResize();

    void resizeEntitiesForCleanup(int additionalCells, int additionalParticles, int additionalTokens);
    void resizeRemainings();
    bool isEmpty();
    void swap();
    void free();

private:
    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities);
};

