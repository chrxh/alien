#include "SimulationData.cuh"

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
    for (int i = 0; i < CellType_Count; ++i) {
        cellTypeOperations[i].init();
    }
}

__device__ void SimulationData::prepareForNextTimestep()
{
    cellMap.reset();
    particleMap.reset();
    processMemory.reset();

    auto maxStructureOperations = 1000 + objects.cellPointers.getNumEntries() / 2; //heuristic
    auto maxCellTypeOperations = objects.cellPointers.getNumEntries();  //heuristic

    structuralOperations.setMemory(processMemory.getTypedSubArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    for (int i = CellType_Base; i < CellType_Count; ++i) {
        cellTypeOperations[i].setMemory(processMemory.getTypedSubArray<CellTypeOperation>(maxCellTypeOperations), maxCellTypeOperations);
    }
    *externalEnergy = cudaSimulationParameters.externalEnergy.value;

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
    return objects.cellPointers.shouldResize_host(cellArraySizeResult * 5)
        || objects.particlePointers.shouldResize_host(particleArraySizeResult * 5)
        || objects.rawMemory.shouldResize_host(additionals.rawMemorySize);
}

void SimulationData::resizeTargetObjects(ArraySizes const& additionals)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, additionals.cellArraySize, additionals.particleArraySize);

    resizeTargetIntern(objects.cellPointers, tempObjects.cellPointers, cellArraySizeResult * 5);
    resizeTargetIntern(objects.particlePointers, tempObjects.particlePointers, particleArraySizeResult * 5);
    resizeTargetIntern(objects.rawMemory, tempObjects.rawMemory, additionals.rawMemorySize);
}

void SimulationData::resizeObjects()
{
    objects.cellPointers.resize(tempObjects.cellPointers.getSize_host());
    objects.particlePointers.resize(tempObjects.particlePointers.getSize_host());
    objects.rawMemory.resize(tempObjects.rawMemory.getSize_host());

    auto estimatedMaxActiveCells = objects.cellPointers.getSize_host() / 5;
    cellMap.resize(estimatedMaxActiveCells);
    auto estimatedMaxActiveParticles = objects.particlePointers.getSize_host() / 5;
    particleMap.resize(estimatedMaxActiveParticles);

    auto upperBoundDynamicMemory =
        (sizeof(StructuralOperation) + sizeof(CellTypeOperation) * CellType_Count + 200) * (estimatedMaxActiveCells + 1000);  //heuristic
    processMemory.resize(upperBoundDynamicMemory);
}

ArraySizes SimulationData::getCurrentArraySizes() const
{
    ArraySizes result;
    result.cellArraySize = objects.cellPointers.getSize_host();
    result.particleArraySize = objects.particlePointers.getSize_host();
    result.rawMemorySize = objects.rawMemory.getSize_host();
    return result;
}

bool SimulationData::isEmpty()
{
    return 0 == objects.rawMemory.getNumEntries_host();
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
    for (int i = 0; i < CellType_Count; ++i) {
        cellTypeOperations[i].free();
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
