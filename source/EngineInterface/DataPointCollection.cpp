#include "DataPointCollection.h"

DataPointCollection DataPointCollection::operator+(DataPointCollection const& other) const
{
    DataPointCollection result;
    result.time = time + other.time;
    result.systemClock = systemClock + other.systemClock;
    result.numCreatures = numCreatures + other.numCreatures;
    result.averageCreatureCells = averageCreatureCells + other.averageCreatureCells;
    result.averageGenomeNodes = averageGenomeNodes + other.averageGenomeNodes;
    result.creatureEnergy = creatureEnergy + other.creatureEnergy;
    result.averageMutationRate = averageMutationRate + other.averageMutationRate;
    result.averageGeneration = averageGeneration + other.averageGeneration;
    result.numLineages = numLineages + other.numLineages;
    result.numSolidObjects = numSolidObjects + other.numSolidObjects;
    result.numFluidObjects = numFluidObjects + other.numFluidObjects;
    result.numCellObjects = numCellObjects + other.numCellObjects;
    result.accumCreatedCreatures = accumCreatedCreatures + other.accumCreatedCreatures;
    result.accumMutations = accumMutations + other.accumMutations;
    result.lineageEntries = other.lineageEntries;  // take the later snapshot
    return result;
}

DataPointCollection DataPointCollection::operator/(double divisor) const
{
    DataPointCollection result;
    result.time = time / divisor;
    result.systemClock = systemClock / divisor;
    result.numCreatures = numCreatures / divisor;
    result.averageCreatureCells = averageCreatureCells / divisor;
    result.averageGenomeNodes = averageGenomeNodes / divisor;
    result.creatureEnergy = creatureEnergy / divisor;
    result.averageMutationRate = averageMutationRate / divisor;
    result.averageGeneration = averageGeneration / divisor;
    result.numLineages = numLineages / divisor;
    result.numSolidObjects = numSolidObjects / divisor;
    result.numFluidObjects = numFluidObjects / divisor;
    result.numCellObjects = numCellObjects / divisor;
    result.accumCreatedCreatures = accumCreatedCreatures / divisor;
    result.accumMutations = accumMutations / divisor;
    result.lineageEntries = lineageEntries;  // pass through unchanged
    return result;
}
