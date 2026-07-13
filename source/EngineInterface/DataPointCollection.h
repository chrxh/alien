#pragma once

#include <EngineInterface/TimelineStatistics.h>

struct DataPoint
{
    double values[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    double summedValues = 0;

    DataPoint operator+(DataPoint const& other) const;
    DataPoint operator/(double divisor) const;
};

struct DataPointCollection
{
    double time;  //could be a time step or real-time
    double systemClock = 0;

    DataPoint numObjects;
    DataPoint numSelfReplicators;
    DataPoint numColonies;
    DataPoint numViruses;
    DataPoint numFreeCells;
    DataPoint numEnergyParticles;
    DataPoint averageGenomeCells;
    DataPoint averageNumCells;
    DataPoint varianceNumCells;
    DataPoint maxNumCellsOfColonies;
    DataPoint totalEnergy;

    DataPoint numCreatedCells;
    DataPoint numAttacks;
    DataPoint numMuscleActivities;
    DataPoint numDefenderActivities;
    DataPoint numDepotActivities;
    DataPoint numInjectionActivities;
    DataPoint numCompletedInjections;
    DataPoint numGeneratorPulses;
    DataPoint numNeuronActivities;
    DataPoint numSensorActivities;
    DataPoint numSensorMatches;
    DataPoint numReconnectorCreated;
    DataPoint numReconnectorRemoved;
    DataPoint numDetonations;

    // Evolution dashboard values (not color-resolved)
    double numCreatures = 0;
    double averageCreatureCells = 0;
    double averageGenomeNodes = 0;
    double creatureEnergy = 0;
    double averageMutationRate = 0;
    double averageGeneration = 0;
    double numLineages = 0;
    double numSolidObjects = 0;
    double numFluidObjects = 0;
    double numCellObjects = 0;
    double accumCreatedCreatures = 0;  // Raw accumulated value; rates are derived GUI-side
    double accumMutations = 0;         // Raw accumulated value; rates are derived GUI-side

    DataPointCollection operator+(DataPointCollection const& other) const;
    DataPointCollection operator/(double divisor) const;
};
