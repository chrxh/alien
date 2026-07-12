#include "DataPointCollection.h"

DataPoint DataPoint::operator+(DataPoint const& other) const
{
    DataPoint result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = values[i] + other.values[i];
    }
    result.summedValues = summedValues + other.summedValues;
    return result;
}

DataPoint DataPoint::operator/(double divisor) const
{
    DataPoint result;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result.values[i] = values[i] / divisor;
    }
    result.summedValues = summedValues / divisor;
    return result;
}

DataPointCollection DataPointCollection::operator+(DataPointCollection const& other) const
{
    DataPointCollection result;
    result.time = time + other.time;
    result.systemClock = systemClock + other.systemClock;
    result.numObjects = numObjects + other.numObjects;
    result.numSelfReplicators = numSelfReplicators + other.numSelfReplicators;
    result.numColonies = numColonies + other.numColonies;
    result.numViruses = numViruses + other.numViruses;
    result.numFreeCells = numFreeCells + other.numFreeCells;
    result.numEnergyParticles = numEnergyParticles + other.numEnergyParticles;
    result.averageGenomeCells = averageGenomeCells + other.averageGenomeCells;
    result.averageNumCells = averageNumCells + other.averageNumCells;
    result.varianceNumCells = varianceNumCells + other.varianceNumCells;
    result.maxNumCellsOfColonies = maxNumCellsOfColonies + other.maxNumCellsOfColonies;
    result.totalEnergy = totalEnergy + other.totalEnergy;
    result.numCreatedCells = numCreatedCells + other.numCreatedCells;
    result.numAttacks = numAttacks + other.numAttacks;
    result.numMuscleActivities = numMuscleActivities + other.numMuscleActivities;
    result.numDefenderActivities = numDefenderActivities + other.numDefenderActivities;
    result.numDepotActivities = numDepotActivities + other.numDepotActivities;
    result.numInjectionActivities = numInjectionActivities + other.numInjectionActivities;
    result.numCompletedInjections = numCompletedInjections + other.numCompletedInjections;
    result.numGeneratorPulses = numGeneratorPulses + other.numGeneratorPulses;
    result.numNeuronActivities = numNeuronActivities + other.numNeuronActivities;
    result.numSensorActivities = numSensorActivities + other.numSensorActivities;
    result.numSensorMatches = numSensorMatches + other.numSensorMatches;
    result.numReconnectorCreated = numReconnectorCreated + other.numReconnectorCreated;
    result.numReconnectorRemoved = numReconnectorRemoved + other.numReconnectorRemoved;
    result.numDetonations = numDetonations + other.numDetonations;
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

DataPointCollection DataPointCollection::operator/(double divisor) const
{
    DataPointCollection result;
    result.time = time / divisor;
    result.systemClock = systemClock / divisor;
    result.numObjects = numObjects / divisor;
    result.numSelfReplicators = numSelfReplicators / divisor;
    result.numColonies = numColonies / divisor;
    result.numViruses = numViruses / divisor;
    result.numFreeCells = numFreeCells / divisor;
    result.numEnergyParticles = numEnergyParticles / divisor;
    result.averageGenomeCells = averageGenomeCells / divisor;
    result.averageNumCells = averageNumCells / divisor;
    result.varianceNumCells = varianceNumCells / divisor;
    result.maxNumCellsOfColonies = maxNumCellsOfColonies / divisor;
    result.totalEnergy = totalEnergy / divisor;
    result.numCreatedCells = numCreatedCells / divisor;
    result.numAttacks = numAttacks / divisor;
    result.numMuscleActivities = numMuscleActivities / divisor;
    result.numDefenderActivities = numDefenderActivities / divisor;
    result.numDepotActivities = numDepotActivities / divisor;
    result.numInjectionActivities = numInjectionActivities / divisor;
    result.numCompletedInjections = numCompletedInjections / divisor;
    result.numGeneratorPulses = numGeneratorPulses / divisor;
    result.numNeuronActivities = numNeuronActivities / divisor;
    result.numSensorActivities = numSensorActivities / divisor;
    result.numSensorMatches = numSensorMatches / divisor;
    result.numReconnectorCreated = numReconnectorCreated / divisor;
    result.numReconnectorRemoved = numReconnectorRemoved / divisor;
    result.numDetonations = numDetonations / divisor;
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
