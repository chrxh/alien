#pragma once

#include <vector>

#include "EngineInterface/Colors.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/StatisticsData.h"

struct DataPoint
{
    double time; //could be a time step or real time

    ColorVector<double> numCellsByColor = {0, 0, 0, 0, 0, 0, 0};

    double numCells = 0;
    double numConnections = 0;
    double numParticles = 0;
    double numCreatedCells = 0;
    double numSuccessfulAttacks = 0;
    double numFailedAttacks = 0;
    double numMuscleActivities = 0;

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
