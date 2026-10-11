#pragma once

#include <atomic>

#include <Data/Interface/CellTypeConstants.h>
#include <Data/Interface/Colors.h>

#include <Engine/Interface/ArraySizesForGpuEntities.h>
#include <Engine/Interface/KernelLaunchSettings.h>

#include "CudaNumberGenerator.cuh"
#include "EnergyParticleGrid.cuh"
#include "ObjectGrid.cuh"
#include "Operations.cuh"
#include "PreprocessedSimulationData.cuh"
#include "SolidGrid.cuh"
#include "WorldGeometry.cuh"

struct SimulationData
{
    // World and grids
    uint64_t* timestep;
    WorldGeometry world;
    ObjectGrid objectGrid;
    EnergyParticleGrid energyParticleGrid;
    SolidGrid solidGrid;

    // Entities
    Entities entities;
    Entities tempEntities;

    // Additional data for cell functions
    double* externalEnergy;
    uint32_t* numConstructorsNeedingEnergyByColor;
    float* externalEnergyInflowPerConstructorByColor;
    PreprocessedSimulationData preprocessedSimulationData;

    // Temporary memory for operations
    Heap processMemory;
    UnmanagedArray<StructuralOperation> structuralOperations;
    UnmanagedArray<CellTypeOperation> cellTypeOperations[CellType_Count];
    UnmanagedArray<int> energyParticlesNearSolids;  // Indices into entities.energies

    // For running gene graph kernels after mutations
    UnmanagedArray<Genome*> mutatedGenomes;

    // Number generators
    CudaNumberGenerator primaryNumberGen;
    CudaNumberGenerator
        secondaryNumberGen;  // Secondary random number generator used in combination with the primary generator for evaluating very low probabilities

    void init(int2 const& worldSize, uint64_t timestep);
    bool shouldResize(ArraySizesForGpuEntities const& sizeDelta);
    void resizeTempObjects(ArraySizesForGpuEntities const& size);
    void resizeObjectsAndTempObjects(ArraySizesForGpuEntities const& size);
    void resizeObjectsByMatchingTempObjects();
    bool isEmpty();
    void free();

private:
    void resizeAuxiliaryData();

    template <typename Entity>
    void resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, uint64_t additionalEntities);
};
