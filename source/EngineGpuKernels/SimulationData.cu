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

namespace
{
    void calcArraySizes(uint64_t& cellArraySizeResult, uint64_t& particleArraySizeResult, uint64_t desiredCellArraySize, uint64_t desiredParticleArraySize)
    {
        auto max = std::max(desiredCellArraySize, desiredParticleArraySize);
        cellArraySizeResult =  desiredCellArraySize * 7 / 10 + max * 3 / 10;
        particleArraySizeResult = desiredParticleArraySize * 7 / 10 + max * 3 / 10;
    }
}

bool SimulationData::shouldResize(ArraySizesForGpu const& sizeDelta)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, sizeDelta.cellArray, sizeDelta.particleArray);
    return objects.cells.shouldResize_host(cellArraySizeResult)
        || objects.particles.shouldResize_host(particleArraySizeResult)
        || objects.heap.shouldResize_host(sizeDelta.heap);
}

void SimulationData::resizeTargetObjects(ArraySizesForGpu const& size)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, size.cellArray, size.particleArray);

    resizeTargetIntern(objects.cells, tempObjects.cells, cellArraySizeResult);
    resizeTargetIntern(objects.particles, tempObjects.particles, particleArraySizeResult);
    resizeTargetIntern(objects.heap, tempObjects.heap, size.heap);
}

void SimulationData::resizeObjects()
{
    objects.cells.resize(tempObjects.cells.getCapacity_host());
    objects.particles.resize(tempObjects.particles.getCapacity_host());
    objects.heap.resize(tempObjects.heap.getCapacity_host());

    auto estimatedMaxActiveCells = objects.cells.getCapacity_host();
    cellMap.resize(estimatedMaxActiveCells);
    auto estimatedMaxActiveParticles = objects.particles.getCapacity_host();
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
        auto newSize = (sourceArray.getCapacity_host() + additionalEntities) * Const::ArrayResizePercentage;
        targetArray.resize(toUInt64(newSize));
    }
}
