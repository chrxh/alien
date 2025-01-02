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
    result.numCells = numCells + other.numCells;
    result.numSelfReplicators = numSelfReplicators + other.numSelfReplicators;
    result.numColonies = numColonies + other.numColonies;
    result.numViruses = numViruses + other.numViruses;
    result.numFreeCells = numFreeCells + other.numFreeCells;
    result.numParticles = numParticles + other.numParticles;
    result.averageGenomeCells = averageGenomeCells + other.averageGenomeCells;
    result.averageGenomeComplexity = averageGenomeComplexity + other.averageGenomeComplexity;
    result.varianceGenomeComplexity = varianceGenomeComplexity + other.varianceGenomeComplexity;
    result.maxGenomeComplexityOfColonies = maxGenomeComplexityOfColonies + other.maxGenomeComplexityOfColonies;
    result.totalEnergy = totalEnergy + other.totalEnergy;
    result.numCreatedCells = numCreatedCells + other.numCreatedCells;
    result.numAttacks = numAttacks + other.numAttacks;
    result.numMuscleActivities = numMuscleActivities + other.numMuscleActivities;
    result.numDefenderActivities = numDefenderActivities + other.numDefenderActivities;
    result.numTransmitterActivities = numTransmitterActivities + other.numTransmitterActivities;
    result.numInjectionActivities = numInjectionActivities + other.numInjectionActivities;
    result.numCompletedInjections = numCompletedInjections + other.numCompletedInjections;
    result.numOscillatorPulses = numOscillatorPulses + other.numOscillatorPulses;
    result.numNeuronActivities = numNeuronActivities + other.numNeuronActivities;
    result.numSensorActivities = numSensorActivities + other.numSensorActivities;
    result.numSensorMatches = numSensorMatches + other.numSensorMatches;
    result.numReconnectorCreated = numReconnectorCreated + other.numReconnectorCreated;
    result.numReconnectorRemoved = numReconnectorRemoved + other.numReconnectorRemoved;
    result.numDetonations = numDetonations + other.numDetonations;
    return result;
}

DataPointCollection DataPointCollection::operator/(double divisor) const
{
    DataPointCollection result;
    result.time = time / divisor;
    result.systemClock = systemClock / divisor;
    result.numCells = numCells / divisor;
    result.numSelfReplicators = numSelfReplicators / divisor;
    result.numColonies = numColonies / divisor;
    result.numViruses = numViruses / divisor;
    result.numFreeCells = numFreeCells / divisor;
    result.numParticles = numParticles / divisor;
    result.averageGenomeCells = averageGenomeCells / divisor;
    result.averageGenomeComplexity = averageGenomeComplexity / divisor;
    result.varianceGenomeComplexity = varianceGenomeComplexity / divisor;
    result.maxGenomeComplexityOfColonies = maxGenomeComplexityOfColonies / divisor;
    result.totalEnergy = totalEnergy / divisor;
    result.numCreatedCells = numCreatedCells / divisor;
    result.numAttacks = numAttacks / divisor;
    result.numMuscleActivities = numMuscleActivities / divisor;
    result.numDefenderActivities = numDefenderActivities / divisor;
    result.numTransmitterActivities = numTransmitterActivities / divisor;
    result.numInjectionActivities = numInjectionActivities / divisor;
    result.numCompletedInjections = numCompletedInjections / divisor;
    result.numOscillatorPulses = numOscillatorPulses / divisor;
    result.numNeuronActivities = numNeuronActivities / divisor;
    result.numSensorActivities = numSensorActivities / divisor;
    result.numSensorMatches = numSensorMatches / divisor;
    result.numReconnectorCreated = numReconnectorCreated / divisor;
    result.numReconnectorRemoved = numReconnectorRemoved / divisor;
    result.numDetonations = numDetonations / divisor;
    return result;
}
