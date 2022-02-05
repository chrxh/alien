#pragma once

#include <atomic>

#include "Base.cuh"
#include "CellFunctionData.cuh"
#include "Definitions.cuh"
#include "EngineInterface/GpuSettings.h"
#include "Entities.cuh"
#include "Map.cuh"
#include "Operations.cuh"
#include "Token.cuh"

struct SimulationData
{
    int2 worldSize;

    CellMap cellMap;
    ParticleMap particleMap;
    CellFunctionData cellFunctionData;

    Entities entities;
    Entities entitiesForCleanup;

    RawMemory processMemory;
    TempArray<StructuralOperation> structuralOperations;
    TempArray<SensorOperation> sensorOperations;

    CudaNumberGenerator numberGen;

    void init(int2 const& worldSize);
    bool shouldResize(int additionalCells, int additionalParticles, int additionalTokens);
    void resizeEntitiesForCleanup(int additionalCells, int additionalParticles, int additionalTokens);
    void resizeRemainings();
    bool isEmpty();
    void free();

    __device__ void prepareForNextTimestep();
    __device__ bool shouldResize();

private:
    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities);
};
