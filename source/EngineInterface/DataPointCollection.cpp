#include "DataPointCollection.h"

OverallDataPoint OverallDataPoint::operator+(OverallDataPoint const& other) const
{
    OverallDataPoint result;
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
    return result;
}

OverallDataPoint OverallDataPoint::operator/(double divisor) const
{
    OverallDataPoint result;
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
    return result;
}

OverallDataPointCollection OverallDataPointCollection::operator+(OverallDataPointCollection const& other) const
{
    OverallDataPointCollection result;
    result.time = time + other.time;
    result.systemClock = systemClock + other.systemClock;
    result.overall = overall + other.overall;
    return result;
}

OverallDataPointCollection OverallDataPointCollection::operator/(double divisor) const
{
    OverallDataPointCollection result;
    result.time = time / divisor;
    result.systemClock = systemClock / divisor;
    result.overall = overall / divisor;
    return result;
}

OverallDataPointCollection DataPointCollection::toOverallDataPointCollection() const
{
    OverallDataPointCollection result;
    result.time = time;
    result.systemClock = systemClock;
    result.overall = overall;
    return result;
}
