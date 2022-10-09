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

    processMemory.init();
    numberGen1.init(40312357);   //some array size for random numbers (~ 40 MB)
    numberGen2.init(1536941);  //some array size for random numbers (~ 1.5 MB)

    structuralOperations.init();
    sensorOperations.init();
    neuralNetOperations.init();
}

__device__ void SimulationData::prepareForNextTimestep()
{
    cellMap.reset();
    particleMap.reset();
    processMemory.reset();

    auto maxStructureOperations = entities.cellPointers.getNumEntries() / 2;
    structuralOperations.setMemory(processMemory.getArray<StructuralOperation>(maxStructureOperations), maxStructureOperations);

    auto maxSensorOperations = entities.cellPointers.getNumEntries() / 2;
    sensorOperations.setMemory(processMemory.getArray<SensorOperation>(maxSensorOperations), maxSensorOperations);

    auto maxNeuralNetOperations = entities.cellPointers.getNumEntries() / 2;
    neuralNetOperations.setMemory(processMemory.getArray<NeuralNetOperation>(maxNeuralNetOperations), maxNeuralNetOperations);

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

    //heuristic
    int upperBoundDynamicMemory = (sizeof(StructuralOperation) + sizeof(NeuralNetOperation) + 200) * (cellArraySize + 1000);
    processMemory.resize(upperBoundDynamicMemory);
}

bool SimulationData::isEmpty()
{
    return 0 == entities.cells.getNumEntries_host() && 0 == entities.particles.getNumEntries_host()
        && 0 == entities.tokens.getNumEntries_host();
}

void SimulationData::free()
{
    entities.free();
    entitiesForCleanup.free();
    cellFunctionData.free();
    cellMap.free();
    particleMap.free();
    numberGen1.free();
    numberGen2.free();
    processMemory.free();

    structuralOperations.free();
    sensorOperations.free();
    neuralNetOperations.free();
}

template <typename Entity>
void SimulationData::resizeTargetIntern(Array<Entity> const& sourceArray, Array<Entity>& targetArray, int additionalEntities)
{
    if (sourceArray.shouldResize_host(additionalEntities)) {
        auto newSize = (sourceArray.getNumEntries_host() + additionalEntities) * 2;
        targetArray.resize(newSize);
    }
}
