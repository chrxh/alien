#include "ConstantMemory.cuh"
#include "GarbageCollectorKernels.cuh"
#include "SimulationData.cuh"

void SimulationData::init(int2 const& worldSize_, uint64_t timestep_)
{
    worldSize = worldSize_;

    entities.init();
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
    return entities.objects.shouldResize_host(cellArraySizeResult) || entities.energyParticles.shouldResize_host(particleArraySizeResult)
        || entities.heap.shouldResize_host(sizeDelta.heap);
}

void SimulationData::resizeTempObjects(ArraySizesForGpu const& size)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, size.cellArray, size.particleArray);

    resizeTargetIntern(entities.objects, tempEntities.objects, cellArraySizeResult);
    resizeTargetIntern(entities.energyParticles, tempEntities.energyParticles, particleArraySizeResult);
    resizeTargetIntern(entities.heap, tempEntities.heap, size.heap);
}

void SimulationData::resizeObjectsAndTempObjects(ArraySizesForGpu const& size)
{
    uint64_t cellArraySizeResult, particleArraySizeResult;
    calcArraySizes(cellArraySizeResult, particleArraySizeResult, size.cellArray, size.particleArray);

    entities.objects.resize(cellArraySizeResult);
    entities.energyParticles.resize(particleArraySizeResult);
    entities.heap.resize(size.heap);
    tempEntities.objects.resize(cellArraySizeResult);
    tempEntities.energyParticles.resize(particleArraySizeResult);
    tempEntities.heap.resize(size.heap);

    resizeAuxiliaryData();
}

void SimulationData::resizeObjectsByMatchingTempObjects()
{
    entities.objects.resize(tempEntities.objects.getCapacity_host());
    entities.energyParticles.resize(tempEntities.energyParticles.getCapacity_host());
    entities.heap.resize(tempEntities.heap.getCapacity_host());

    resizeAuxiliaryData();
}

bool SimulationData::isEmpty()
{
    return 0 == entities.heap.getNumEntries_host();
}

void SimulationData::free()
{
    entities.free();
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
    auto estimatedMaxActiveCells = entities.objects.getCapacity_host();
    cellMap.resize(estimatedMaxActiveCells);
    auto estimatedMaxActiveParticles = entities.energyParticles.getCapacity_host();
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
