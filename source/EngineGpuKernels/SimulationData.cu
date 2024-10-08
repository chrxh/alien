﻿#include "SimulationData.cuh"

#include "ConstantMemory.cuh"
#include "GarbageCollectorKernels.cuh"

void SimulationData::init(int2 const& worldSize_, uint64_t timestep_)
{
    worldSize = worldSize_;
    timestep = timestep_;

    objects.init();
    tempObjects.init();
    preprocessedSimulationData.init(worldSize);
    cellMap.init(worldSize);
    particleMap.init(worldSize);

    CudaMemoryManager::getInstance().acquireMemory<double>(1, externalEnergy);
    CHECK_FOR_CUDA_ERROR(cudaMemset(externalEnergy, 0, sizeof(double)));
 
    processMemory.init();
    numberGen1.init(40312357);   //some array size for random numbers (~ 40 MB)
    numberGen2.init(1536941);  //some array size for random numbers (~ 1.5 MB)

    structuralOperations.init();
    for (int i = 0; i < CellFunction_WithoutNone_Count; ++i) {
        cellFunctionOperations[i].init();
    }
}

__device__ void SimulationData::prepareForNextTimestep()
{
    cellMap.reset();
    particleMap.reset();
    processMemory.reset();

    auto maxStructureOperations = 1000 + objects.cellPointers.getNumEntries() / 2; //heuristic
    auto maxCellFunctionOperations = objects.cellPointers.getNumEntries();  //heuristic

    structuralOperations.setMemory(processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    for (int i = 0; i < CellFunction_WithoutNone_Count; ++i) {
        cellFunctionOperations[i].setMemory(processMemory.getTypedSubArray<CellFunctionOperation>(maxCellFunctionOperations), maxCellFunctionOperations);
    }
    *externalEnergy = cudaSimulationParameters.externalEnergy;

    objects.saveNumEntries();
}

namespace
{
    void calcArraySizes(uint64_t& cellArraySizeResult, uint64_t& particleArraySizeResult, uint64_t desiredCellArraySize, uint64_t desiredParticleArraySize)
    {
        auto max = std::max(desiredCellArraySize, desiredParticleArraySize);
        cellArraySizeResult =  desiredCellArraySize * 7 / 10 + max * 3 / 10;
        particleArraySizeResult = desiredParticleArraySize * 7 / 10 + max * 3 / 10;
    }
}

bool SimulationData::shouldResize(ArraySizes const& additionals)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, additionals.cellArraySize, additionals.particleArraySize);
    return objects.cells.shouldResize_host(cellArraySizeResult) || objects.cellPointers.shouldResize_host(cellArraySizeResult * 5)
        || objects.particles.shouldResize_host(particleArraySizeResult) || objects.particlePointers.shouldResize_host(particleArraySizeResult * 5)
        || objects.auxiliaryData.shouldResize_host(additionals.auxiliaryDataSize);
}

void SimulationData::resizeTargetObjects(ArraySizes const& additionals)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, additionals.cellArraySize, additionals.particleArraySize);

    resizeTargetIntern(objects.cells, tempObjects.cells, cellArraySizeResult);
    resizeTargetIntern(objects.cellPointers, tempObjects.cellPointers, cellArraySizeResult * 5);
    resizeTargetIntern(objects.particles, tempObjects.particles, particleArraySizeResult);
    resizeTargetIntern(objects.particlePointers, tempObjects.particlePointers, particleArraySizeResult * 5);
    resizeTargetIntern(objects.auxiliaryData, tempObjects.auxiliaryData, additionals.auxiliaryDataSize);
}

void SimulationData::resizeObjects()
{
    objects.cells.resize(tempObjects.cells.getSize_host());
    objects.cellPointers.resize(tempObjects.cellPointers.getSize_host());
    objects.particles.resize(tempObjects.particles.getSize_host());
    objects.particlePointers.resize(tempObjects.particlePointers.getSize_host());
    objects.auxiliaryData.resize(tempObjects.auxiliaryData.getSize_host());

    auto cellArraySize = objects.cells.getSize_host();
    cellMap.resize(cellArraySize);
    auto particleArraySize = objects.particles.getSize_host();
    particleMap.resize(particleArraySize);

    int upperBoundDynamicMemory = (sizeof(StructuralOperation) + sizeof(CellFunctionOperation) * CellFunction_Count + 200) * (cellArraySize + 1000); //heuristic
    processMemory.resize(upperBoundDynamicMemory);
}

bool SimulationData::isEmpty()
{
    return 0 == objects.cells.getNumEntries_host() && 0 == objects.particles.getNumEntries_host();
}

void SimulationData::free()
{
    objects.free();
    tempObjects.free();
    preprocessedSimulationData.free();
    cellMap.free();
    particleMap.free();
    numberGen1.free();
    numberGen2.free();
    processMemory.free();
    CudaMemoryManager::getInstance().freeMemory(externalEnergy);

    structuralOperations.free();
    for (int i = 0; i < CellFunction_WithoutNone_Count; ++i) {
        cellFunctionOperations[i].free();
    }
}

template <typename Entity>
void SimulationData::resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, uint64_t additionalEntities)
{
    if (sourceArray.shouldResize_host(additionalEntities)) {
        auto newSize = (sourceArray.getSize_host() + additionalEntities) * 3;
        targetArray.resize(newSize);
    }
}
