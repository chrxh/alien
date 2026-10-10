#include "DataPointCollection.h"

ColorOverallDataPoint ColorOverallDataPoint::operator+(ColorOverallDataPoint const& other) const
{
    ColorOverallDataPoint result;
    result.numCreatures = numCreatures + other.numCreatures;
    result.numGenomes = numGenomes + other.numGenomes;
    result.sumCreatureCells = sumCreatureCells + other.sumCreatureCells;
    result.sumCreatureGenerations = sumCreatureGenerations + other.sumCreatureGenerations;
    result.sumGenomeNodes = sumGenomeNodes + other.sumGenomeNodes;
    result.sumMutationRates = sumMutationRates + other.sumMutationRates;
    result.sumCreatureEnergy = sumCreatureEnergy + other.sumCreatureEnergy;
    result.numCreatedCreatures = numCreatedCreatures + other.numCreatedCreatures;
    result.totalMutations = totalMutations + other.totalMutations;
    result.totalAttackedEnergy = totalAttackedEnergy + other.totalAttackedEnergy;
    result.totalMuscleActivity = totalMuscleActivity + other.totalMuscleActivity;
    return result;
}

ColorOverallDataPoint ColorOverallDataPoint::operator/(double divisor) const
{
    ColorOverallDataPoint result;
    result.numCreatures = numCreatures / divisor;
    result.numGenomes = numGenomes / divisor;
    result.sumCreatureCells = sumCreatureCells / divisor;
    result.sumCreatureGenerations = sumCreatureGenerations / divisor;
    result.sumGenomeNodes = sumGenomeNodes / divisor;
    result.sumMutationRates = sumMutationRates / divisor;
    result.sumCreatureEnergy = sumCreatureEnergy / divisor;
    result.numCreatedCreatures = numCreatedCreatures / divisor;
    result.totalMutations = totalMutations / divisor;
    result.totalAttackedEnergy = totalAttackedEnergy / divisor;
    result.totalMuscleActivity = totalMuscleActivity / divisor;
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
    result.totalAttackedEnergy = totalAttackedEnergy + other.totalAttackedEnergy;
    result.totalMuscleActivity = totalMuscleActivity + other.totalMuscleActivity;
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
    result.totalAttackedEnergy = totalAttackedEnergy / divisor;
    result.totalMuscleActivity = totalMuscleActivity / divisor;
    return result;
}

DataPointCollection DataPointCollection::operator+(DataPointCollection const& other) const
{
    DataPointCollection result;
    result.timestep = timestep + other.timestep;
    result.systemClock = systemClock + other.systemClock;
    result.overall = overall;
    for (auto const& [colorBitset, dataPoint] : other.overall) {
        auto it = result.overall.find(colorBitset);
        if (it != result.overall.end()) {
            it->second = it->second + dataPoint;
        } else {
            result.overall.emplace(colorBitset, dataPoint);
        }
    }
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
    result.timestep = timestep / divisor;
    result.systemClock = systemClock / divisor;
    result.overall = overall;
    for (auto& [colorBitset, dataPoint] : result.overall) {
        dataPoint = dataPoint / divisor;
    }
    result.lineages = lineages;
    for (auto& [lineageId, dataPoint] : result.lineages) {
        dataPoint = dataPoint / divisor;
    }
    return result;
}
