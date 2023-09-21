#pragma once

#include <vector>

#include "EngineInterface/Colors.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/StatisticsData.h"

struct DataPoint
{
    double values[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    double summedValues = 0;

    DataPoint operator+(DataPoint const& other) const;
    DataPoint operator/(double divisor) const;
};

struct DataPointCollection
{
    double time; //could be a time step or real time

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

    DataPointCollection operator+(DataPointCollection const& other) const;
    DataPointCollection operator/(double divisor) const;
};

struct TimelineLiveStatistics
{
    static double constexpr MaxLiveHistory = 240.0f;  //in seconds

    double timepoint = 0.0f;  //in seconds
    float history = 10.0f;   //in seconds

    std::vector<DataPointCollection> dataPointCollectionHistory;
    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;

    void truncate();
    void add(TimelineStatistics const& statistics, uint64_t timestep);
};

struct TimelineLongtermStatistics
{
    double longtermTimestepDelta = 10.0f;

    std::vector<DataPointCollection> dataPointCollectionHistory;
    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;

    void add(TimelineStatistics const& statistics, uint64_t timestep);
};
