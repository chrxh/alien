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

    // Heuristics
    auto maxStructureOperations = 1000 + objects.cellPointers.getNumEntries() / 2;
    auto maxCellTypeOperations = objects.cellPointers.getNumEntries();

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

bool SimulationData::shouldResize(ArraySizesForObjects const& sizeDelta)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, sizeDelta.cellArray, sizeDelta.particleArray);
    return objects.cellPointers.shouldResize_host(cellArraySizeResult)
        || objects.particlePointers.shouldResize_host(particleArraySizeResult)
        || objects.heap.shouldResize_host(sizeDelta.heap);
}

void SimulationData::resizeTargetObjects(ArraySizesForObjects const& size)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, size.cellArray, size.particleArray);

    resizeTargetIntern(objects.cellPointers, tempObjects.cellPointers, cellArraySizeResult);
    resizeTargetIntern(objects.particlePointers, tempObjects.particlePointers, particleArraySizeResult);
    resizeTargetIntern(objects.heap, tempObjects.heap, size.heap);
}

void SimulationData::resizeObjects()
{
    objects.cellPointers.resize(tempObjects.cellPointers.getSize_host());
    objects.particlePointers.resize(tempObjects.particlePointers.getSize_host());
    objects.heap.resize(tempObjects.heap.getSize_host());

    auto estimatedMaxActiveCells = objects.cellPointers.getSize_host();
    cellMap.resize(estimatedMaxActiveCells);
    auto estimatedMaxActiveParticles = objects.particlePointers.getSize_host();
    particleMap.resize(estimatedMaxActiveParticles);

    auto upperBoundDynamicMemory =
        (sizeof(StructuralOperation) + sizeof(CellTypeOperation) * CellType_Count + 200) * (estimatedMaxActiveCells + 1000);  // Heuristic
    processMemory.resize(upperBoundDynamicMemory);
}

bool SimulationData::isEmpty()
{
    return 0 == objects.heap.getNumEntries_host();
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
        auto newSize = (sourceArray.getSize_host() + additionalEntities) * Const::ArrayResizePercentage;
        targetArray.resize(toUInt64(newSize));
    }
}
