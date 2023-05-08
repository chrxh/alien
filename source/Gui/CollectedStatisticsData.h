#pragma once

#include <vector>

#include "EngineInterface/Colors.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/StatisticsData.h"

struct DataPoint
{
    double time; //could be a time step or real time

    ColorVector<double> numCells = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numSelfReplicators = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numViruses = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numConnections = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numParticles = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> totalEnergy = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numCreatedCells = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numAttacks = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numMuscleActivities = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numDefenderActivities = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numTransmitterActivities = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numInjectionActivities = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numCompletedInjections = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numNervePulses = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numNeuronActivities = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numSensorActivities = {0, 0, 0, 0, 0, 0, 0};
    ColorVector<double> numSensorMatches = {0, 0, 0, 0, 0, 0, 0};

    DataPoint operator+(DataPoint const& other) const;
    DataPoint operator/(double divisor) const;
};

struct TimelineLiveStatistics
{
    static double constexpr MaxLiveHistory = 120.0f;  //in seconds

    double timepoint = 0.0f;  //in seconds
    float history = 10.0f;   //in seconds

    std::vector<DataPoint> dataPoints;
    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;

    void truncate();
    void add(TimelineStatistics const& statistics, uint64_t timestep);
};

struct TimelineLongtermStatistics
{
    double longtermTimestepDelta = 10.0f;

    std::vector<DataPoint> dataPoints;
    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;

    void add(TimelineStatistics const& statistics, uint64_t timestep);
};
