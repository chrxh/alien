#pragma once

#include <atomic>

#include "EngineInterface/Enums.h"
#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "ProprocessedCellFunctionData.cuh"
#include "Definitions.cuh"
#include "Objects.cuh"
#include "Map.cuh"
#include "Operations.cuh"

struct SimulationData
{
    //maps
    uint64_t timestep;
    int2 worldSize;
    CellMap cellMap;
    ParticleMap particleMap;

    //objects
    Objects objects;
    Objects tempObjects;

    //additional data for cell functions
    RawMemory processMemory;
    PreprocessedCellFunctionData preprocessedCellFunctionData;

    //scheduled operations
    TempArray<StructuralOperation> structuralOperations;
    TempArray<CellFunctionOperation> cellFunctionOperations[Enums::CellFunction_WithoutNoneCount];

    //number generators
    CudaNumberGenerator numberGen1;
    CudaNumberGenerator numberGen2;  //second random number generator used in combination with the first generator for evaluating very low probabilities

    void init(int2 const& worldSize, uint64_t timestep);
    bool shouldResize(int additionalCells, int additionalParticles);
    void resizeEntitiesForCleanup(int additionalCells, int additionalParticles);
    void resizeRemainings();
    bool isEmpty();
    void free();

    __device__ void prepareForNextTimestep();
    __device__ bool shouldResize();

private:
    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities);
};
