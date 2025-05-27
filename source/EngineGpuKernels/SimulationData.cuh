#pragma once

#include <atomic>

#include "EngineInterface/ArraySizesForGpu.h"
#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/GpuSettings.h"

#include "Base.cuh"
#include "CudaNumberGenerator.cuh"
#include "PreprocessedSimulationData.cuh"
#include "Definitions.cuh"
#include "Objects.cuh"
#include "Map.cuh"
#include "Operations.cuh"

struct SimulationData
{
    // Maps
    uint64_t timestep;
    int2 worldSize;
    CellMap cellMap;
    ParticleMap particleMap;

    // Objects
    Objects objects;
    Objects tempObjects;

    // Additional data for cell functions
    double* externalEnergy;
    Heap processMemory;
    PreprocessedSimulationData preprocessedSimulationData;

    // Temporary memory for operations
    UnmanagedArray<StructuralOperation> structuralOperations;
    UnmanagedArray<CellTypeOperation> cellTypeOperations[CellType_Count];

    // Number generators
    CudaNumberGenerator primaryNumberGen;
    CudaNumberGenerator secondaryNumberGen;  // Secondary random number generator used in combination with the primary generator for evaluating very low probabilities

    void init(int2 const& worldSize, uint64_t timestep);
    bool shouldResize(ArraySizesForGpu const& sizeDelta);
    void resizeTargetObjects(ArraySizesForGpu const& size);
    void resizeObjects();
    bool isEmpty();
    void free();

private:
    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, uint64_t additionalEntities);
};
