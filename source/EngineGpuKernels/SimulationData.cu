#include "SimulationData.cuh"

#include "Token.cuh"
#include "GarbageCollectorKernels.cuh"

void SimulationData::init(int2 const& worldSize_)
{
    worldSize = worldSize_;

    entities.init();
    entitiesForCleanup.init();
    cellFunctionData.init(worldSize);
    cellMap.init(worldSize);
    particleMap.init(worldSize);

    tempMemory.init();
    numberGen.init(40312357);   //some array size for random numbers (~ 40 MB)

    structuralOperations.init();
    sensorOperations.init();
}

__device__ void SimulationData::prepareForNextTimestep()
{
    cellMap.reset();
    particleMap.reset();
    tempMemory.reset();

    auto maxStructureOperations = entities.cellPointers.getNumEntries() / 2;
    structuralOperations.setMemory(tempMemory.getArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    auto maxSensorOperations = entities.cellPointers.getNumEntries() / 2;
    sensorOperations.setMemory(tempMemory.getArray<SensorOperation>(maxSensorOperations), maxSensorOperations);

    entities.saveNumEntries();
}

bool SimulationData::shouldResize(int additionalCells, int additionalParticles, int additionalTokens)
{
    auto cellAndParticleArraySizeInc = std::max(additionalCells, additionalParticles);
    auto tokenArraySizeInc = std::max(additionalTokens, cellAndParticleArraySizeInc / 3);

    return entities.cells.shouldResize_host(cellAndParticleArraySizeInc)
        || entities.cellPointers.shouldResize_host(cellAndParticleArraySizeInc * 10)
        || entities.particles.shouldResize_host(cellAndParticleArraySizeInc)
        || entities.particlePointers.shouldResize_host(cellAndParticleArraySizeInc * 10)
        || entities.tokens.shouldResize_host(tokenArraySizeInc)
        || entities.tokenPointers.shouldResize_host(tokenArraySizeInc * 10);
}

__device__ bool SimulationData::shouldResize()
{
    return entities.cells.shouldResize(0) || entities.cellPointers.shouldResize(0)
        || entities.particles.shouldResize(0) || entities.particlePointers.shouldResize(0)
        || entities.tokens.shouldResize(0) || entities.tokenPointers.shouldResize(0);
}

void SimulationData::resizeEntitiesForCleanup(int additionalCells, int additionalParticles, int additionalTokens)
{
    auto cellAndParticleArraySizeInc = std::max(additionalCells, additionalParticles);
    auto tokenArraySizeInc = std::max(additionalTokens, cellAndParticleArraySizeInc / 3);

    resizeTargetIntern(entities.cells, entitiesForCleanup.cells, cellAndParticleArraySizeInc);
    resizeTargetIntern(entities.cellPointers, entitiesForCleanup.cellPointers, cellAndParticleArraySizeInc * 10);
    resizeTargetIntern(entities.particles, entitiesForCleanup.particles, cellAndParticleArraySizeInc);
    resizeTargetIntern(entities.particlePointers, entitiesForCleanup.particlePointers, cellAndParticleArraySizeInc * 10);
    resizeTargetIntern(entities.tokens, entitiesForCleanup.tokens, tokenArraySizeInc);
    resizeTargetIntern(entities.tokenPointers, entitiesForCleanup.tokenPointers, tokenArraySizeInc * 10);
    printf("SIZE: %d, OLD: %llu \n", entitiesForCleanup.cells.getSize_host(), entitiesForCleanup.dynamicMemory.getSize());
    KERNEL_CALL_1_1(cudaCleanupStringBytes2, entities.cellPointers);
    entitiesForCleanup.dynamicMemory.resize(entitiesForCleanup.cells.getSize_host() * STRING_BYTES_PER_CELL);
    KERNEL_CALL_1_1(cudaCleanupStringBytes2, entities.cellPointers);
}

void SimulationData::resizeRemainings()
{
    entities.cells.resize(entitiesForCleanup.cells.getSize_host());
    entities.cellPointers.resize(entitiesForCleanup.cellPointers.getSize_host());
    entities.particles.resize(entitiesForCleanup.particles.getSize_host());
    entities.particlePointers.resize(entitiesForCleanup.particlePointers.getSize_host());
    entities.tokens.resize(entitiesForCleanup.tokens.getSize_host());
    entities.tokenPointers.resize(entitiesForCleanup.tokenPointers.getSize_host());

    auto cellArraySize = entities.cells.getSize_host();
    cellMap.resize(cellArraySize);
    particleMap.resize(cellArraySize);

    printf("SIZE2: %llu \n", entitiesForCleanup.dynamicMemory.getSize());
    entities.dynamicMemory.resize(entitiesForCleanup.dynamicMemory.getSize() * STRING_BYTES_PER_CELL);

    //heuristic
    int upperBoundDynamicMemory = (sizeof(StructuralOperation) + 200) * (cellArraySize + 1000);
    tempMemory.resize(upperBoundDynamicMemory);
}

bool SimulationData::isEmpty()
{
    return 0 == entities.cells.getNumEntries_host() && 0 == entities.particles.getNumEntries_host()
        && 0 == entities.tokens.getNumEntries_host();
}

void SimulationData::swap()
{
    entities.cells.swapContent_host(entitiesForCleanup.cells);
    entities.cellPointers.swapContent_host(entitiesForCleanup.cellPointers);
    entities.particles.swapContent_host(entitiesForCleanup.particles);

    entities.particlePointers.swapContent_host(entitiesForCleanup.particlePointers);
    entities.tokens.swapContent_host(entitiesForCleanup.tokens);
    entities.tokenPointers.swapContent_host(entitiesForCleanup.tokenPointers);
    entities.dynamicMemory.swapContent_host(entitiesForCleanup.dynamicMemory);
}

void SimulationData::free()
{
    entities.free();
    entitiesForCleanup.free();
    cellFunctionData.free();
    cellMap.free();
    particleMap.free();
    numberGen.free();
    tempMemory.free();

    structuralOperations.free();
    sensorOperations.free();
}

template <typename Entity>
void SimulationData::resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities)
{
    if (sourceArray.shouldResize_host(additionalEntities)) {
        auto newSize = (sourceArray.getNumEntries_host() + additionalEntities) * 2;
        targetArray.resize(newSize);
    }
}
