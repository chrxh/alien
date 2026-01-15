#include "ConstantMemory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "SimulationData.cuh"

void SimulationData::init(int2 const& worldSize_, uint64_t timestep_)
{
    worldSize = worldSize_;

    objects.init();
    tempEntities.init();
    preprocessedSimulationData.init(worldSize);
    cellMap.init(worldSize);
    particleMap.init(worldSize);

    CudaMemoryManager::getInstance().acquireMemory<double>(1, externalEnergy);
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, timestep);
    copyToDevice(timestep, &timestep_);
    CHECK_FOR_CUDA_ERROR(cudaMemset(externalEnergy, 0, sizeof(double)));


    processMemory.init();
    primaryNumberGen.init(40312357);   //some array size for random numbers (~ 160 MB)
    secondaryNumberGen.init(1536941);  //some array size for random numbers (~ 6 MB)

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
        cellArraySizeResult = desiredCellArraySize * 7 / 10 + max * 3 / 10;
        particleArraySizeResult = desiredParticleArraySize * 7 / 10 + max * 3 / 10;
    }
}

bool SimulationData::shouldResize(ArraySizesForGpu const& sizeDelta)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, sizeDelta.cellArray, sizeDelta.particleArray);
    return objects.entities.shouldResize_host(cellArraySizeResult) || objects.energyParticles.shouldResize_host(particleArraySizeResult)
        || objects.heap.shouldResize_host(sizeDelta.heap);
}

void SimulationData::resizeTempObjects(ArraySizesForGpu const& size)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, size.cellArray, size.particleArray);

    resizeTargetIntern(objects.objects, tempEntities.objects, cellArraySizeResult);
    resizeTargetIntern(objects.energyParticles, tempEntities.energyParticles, particleArraySizeResult);
    resizeTargetIntern(objects.heap, tempEntities.heap, size.heap);
}

void SimulationData::resizeObjectsAndTempObjects(ArraySizesForGpu const& size)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, size.cellArray, size.particleArray);

    objects.entities.resize(cellArraySizeResult);
    objects.energyParticles.resize(particleArraySizeResult);
    objects.heap.resize(size.heap);
    tempEntities.entities.resize(cellArraySizeResult);
    tempEntities.energyParticles.resize(particleArraySizeResult);
    tempEntities.heap.resize(size.heap);

    resizeAuxiliaryData();
}

void SimulationData::resizeObjectsByMatchingTempObjects()
{
    objects.entities.resize(tempEntities.entities.getCapacity_host());
    objects.energyParticles.resize(tempEntities.energyParticles.getCapacity_host());
    objects.heap.resize(tempEntities.heap.getCapacity_host());

    resizeAuxiliaryData();
}

bool SimulationData::isEmpty()
{
    return 0 == objects.heap.getNumEntries_host();
}

void SimulationData::free()
{
    objects.free();
    tempEntities.free();
    preprocessedSimulationData.free();
    cellMap.free();
    particleMap.free();
    primaryNumberGen.free();
    secondaryNumberGen.free();
    processMemory.free();
    CudaMemoryManager::getInstance().freeMemory(externalEnergy);
    CudaMemoryManager::getInstance().freeMemory(timestep);

    structuralOperations.free();
    for (int i = 0; i < CellType_Count; ++i) {
        cellTypeOperations[i].free();
    }
}

void SimulationData::resizeAuxiliaryData()
{
    auto estimatedMaxActiveCells = objects.entities.getCapacity_host();
    cellMap.resize(estimatedMaxActiveCells);
    auto estimatedMaxActiveParticles = objects.energyParticles.getCapacity_host();
    particleMap.resize(estimatedMaxActiveParticles);

    auto upperBoundDynamicMemory =
        (sizeof(StructuralOperation) + sizeof(CellTypeOperation) * CellType_Count + 200) * (estimatedMaxActiveCells + 1000);  // Heuristic
    processMemory.resize(upperBoundDynamicMemory);
}

template <typename Entity>
void SimulationData::resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, uint64_t additionalEntities)
{
    if (sourceArray.shouldResize_host(additionalEntities)) {
        auto newSize = (sourceArray.getCapacity_host() + additionalEntities) * Const::ArrayResizePercentage;
        targetArray.resize(toUInt64(newSize));
    }
}
