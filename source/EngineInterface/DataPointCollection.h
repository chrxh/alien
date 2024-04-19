#pragma once

#include "EngineInterface/RawStatisticsData.h"

struct DataPoint
{
    double values[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    double summedValues = 0;

    DataPoint operator+(DataPoint const& other) const;
    DataPoint operator/(double divisor) const;
};

struct DataPointCollection
{
    double time; //could be a time step or real-time

    DataPoint numCells;
    DataPoint numSelfReplicators;
    DataPoint numViruses;
    DataPoint numConnections;
    DataPoint numParticles;
    DataPoint averageGenomeCells;
    DataPoint totalEnergy;

    DataPoint numCreatedCells;
    DataPoint numAttacks;
    DataPoint numMuscleActivities;
    DataPoint numDefenderActivities;
    DataPoint numTransmitterActivities;
    DataPoint numInjectionActivities;
    DataPoint numCompletedInjections;
    DataPoint numNervePulses;
    DataPoint numNeuronActivities;
    DataPoint numSensorActivities;
    DataPoint numSensorMatches;
    DataPoint numReconnectorCreated;
    DataPoint numReconnectorRemoved;
    DataPoint numDetonations;

    DataPointCollection operator+(DataPointCollection const& other) const;
    DataPointCollection operator/(double divisor) const;
};
