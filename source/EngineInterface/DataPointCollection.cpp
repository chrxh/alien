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
    result.numEnergyParticles = numEnergyParticles + other.numEnergyParticles;
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
    result.numEnergyParticles = numEnergyParticles / divisor;
    result.accumCreatedCreatures = accumCreatedCreatures / divisor;
    result.accumMutations = accumMutations / divisor;
    return result;
}

LineageDataPoint LineageDataPoint::operator+(LineageDataPoint const& other) const
{
    LineageDataPoint result;
    result.colorBitset = colorBitset | other.colorBitset;
    result.representativeCellId = representativeCellId != 0 ? representativeCellId : other.representativeCellId;
    result.numCreatures = numCreatures + other.numCreatures;
    result.numGenomes = numGenomes + other.numGenomes;
    result.sumCreatureCells = sumCreatureCells + other.sumCreatureCells;
    result.sumCreatureGenerations = sumCreatureGenerations + other.sumCreatureGenerations;
    result.sumGenomeNodes = sumGenomeNodes + other.sumGenomeNodes;
    result.sumMutationRates = sumMutationRates + other.sumMutationRates;
    result.sumCreatureEnergy = sumCreatureEnergy + other.sumCreatureEnergy;
    result.numCreatedCreatures = numCreatedCreatures + other.numCreatedCreatures;
    result.totalMutations = totalMutations + other.totalMutations;
    return result;
}

LineageDataPoint LineageDataPoint::operator/(double divisor) const
{
    LineageDataPoint result;
    result.colorBitset = colorBitset;
    result.representativeCellId = representativeCellId;
    result.numCreatures = numCreatures / divisor;
    result.numGenomes = numGenomes / divisor;
    result.sumCreatureCells = sumCreatureCells / divisor;
    result.sumCreatureGenerations = sumCreatureGenerations / divisor;
    result.sumGenomeNodes = sumGenomeNodes / divisor;
    result.sumMutationRates = sumMutationRates / divisor;
    result.sumCreatureEnergy = sumCreatureEnergy / divisor;
    result.numCreatedCreatures = numCreatedCreatures / divisor;
    result.totalMutations = totalMutations / divisor;
    return result;
}

DataPointCollection DataPointCollection::operator+(DataPointCollection const& other) const
{
    DataPointCollection result;
    result.time = time + other.time;
    result.timestep = timestep + other.timestep;
    result.systemClock = systemClock + other.systemClock;
    result.overall = overall + other.overall;
    result.lineages = lineages;
    for (auto const& [lineageId, dataPoint] : other.lineages) {
        auto it = result.lineages.find(lineageId);
        if (it != result.lineages.end()) {
            it->second = it->second + dataPoint;
        } else {
            result.lineages.emplace(lineageId, dataPoint);
        }
    }
    return result;
}

DataPointCollection DataPointCollection::operator/(double divisor) const
{
    DataPointCollection result;
    result.time = time / divisor;
    result.timestep = timestep / divisor;
    result.systemClock = systemClock / divisor;
    result.overall = overall / divisor;
    result.lineages = lineages;
    for (auto& [lineageId, dataPoint] : result.lineages) {
        dataPoint = dataPoint / divisor;
    }
    return result;
}
