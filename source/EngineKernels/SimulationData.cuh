#pragma once

#include <atomic>

#include <Data/CellTypeConstants.h>
#include <Data/Colors.h>

#include <EngineInterface/ArraySizesForGpuEntities.h>
#include <EngineInterface/KernelLaunchSettings.h>

#include "BarrierGrid.cuh"
#include "CudaNumberGenerator.cuh"
#include "DomainContext.cuh"
#include "DomainOps.cuh"
#include "EnergyParticleGrid.cuh"
#include "ObjectGrid.cuh"
#include "Operations.cuh"
#include "PreprocessedSimulationData.cuh"
#include "SensorScans.cuh"
#include "WorldGeometry.cuh"

enum ExternalEnergyDemand_
{
    ExternalEnergyDemand_Sources,
    ExternalEnergyDemand_Constructors,
    ExternalEnergyDemand_Count
};

struct SimulationData
{
    // Domain decomposition
    DomainContext domain;
    Array<DomainOp> domainOps;                            // Changes to ghosts, sent to their owners in the next sync round
    Array<DomainOp> receivedShockWaves;                   // Applied in the next time step, when the object grid is filled
    Array<SensorContinuation> sensorContinuations;        // Scans whose rays continue in other strips
    Array<SensorScanRequest> sensorScanRequests;          // Sent in the next sync round
    Array<SensorScanRequest> receivedSensorScanRequests;  // Scanned in the next time step
    Array<SensorScanResponse> sensorScanResponses;        // Sent in the next sync round
    Array<PendingSensorScan> pendingSensorScans[2];       // Indexed by the parity of the time step in which the scans started

    // World and grids
    uint64_t* timestep;
    WorldGeometry world;
    ObjectGrid objectGrid;
    EnergyParticleGrid energyParticleGrid;
    BarrierGrid barrierGrid;

    // Entities
    Entities entities;
    Entities tempEntities;

    // Additional data for cell functions
    double* externalEnergy;
    double* externalEnergyDemands;  // Requested in the current time step, only counted in a decomposed simulation
    uint32_t* numConstructorsNeedingEnergyByColor;
    float* externalEnergyInflowPerConstructorByColor;
    PreprocessedSimulationData preprocessedSimulationData;

    // Temporary memory for operations
    Heap processMemory;
    UnmanagedArray<StructuralOperation> structuralOperations;
    UnmanagedArray<CellTypeOperation> cellTypeOperations[CellType_Count];
    UnmanagedArray<int> energyParticlesNearBarriers;  // Indices into entities.energies

    // For running gene graph kernels after mutations
    UnmanagedArray<Genome*> mutatedGenomes;

    // Number generators
    CudaNumberGenerator primaryNumberGen;
    CudaNumberGenerator
        secondaryNumberGen;  // Secondary random number generator used in combination with the primary generator for evaluating very low probabilities

    void init(int2 const& worldSize, uint64_t timestep);
    bool shouldResize(ArraySizesForGpuEntities const& sizeDelta);
    static bool shouldResize(ArraySizesForGpuEntities const& sizeDelta, ArraySizesForGpuEntities const& numEntries, ArraySizesForGpuEntities const& capacities);
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
