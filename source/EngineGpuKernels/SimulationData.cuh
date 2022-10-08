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
    //maps
    int2 worldSize;
    CellMap cellMap;
    ParticleMap particleMap;

    //objects
    Entities entities;
    Entities entitiesForCleanup;

    //additional data for cell functions
    RawMemory processMemory;
    CellFunctionData cellFunctionData;

    //scheduled operations
    TempArray<StructuralOperation> structuralOperations;
    TempArray<SensorOperation> sensorOperations;
    TempArray<NeuralNetOperation> neuralNetOperations;

    //number generators
    CudaNumberGenerator numberGen1;
    CudaNumberGenerator numberGen2;  //second random number generator used in combination with the first generator for evaluating very low probabilities

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
