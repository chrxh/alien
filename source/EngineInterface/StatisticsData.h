#pragma once

#include "EngineInterface/Constants.h"
#include "EngineInterface/Colors.h"


struct TimestepStatistics
{
    ColorVector<int> numCellsByColor = {0, 0, 0, 0, 0, 0, 0};
    int numConnections = 0;
    int numParticles = 0;
};

struct AccumulatedStatistics
{
    uint64_t numCreatedCells = 0;
    uint64_t numSuccessfulAttacks = 0;
    uint64_t numFailedAttacks = 0;
    uint64_t numMuscleActivities = 0;
};

struct TimelineStatistics
{
    TimestepStatistics timestep;
    AccumulatedStatistics accumulated;
};

struct HistogramData
{
    int maxValue = 0;
    int numCellsByColorBySlot[MAX_COLORS][MAX_HISTOGRAM_SLOTS];
};

struct StatisticsData
{
    TimelineStatistics timeline;
    HistogramData histogram;
};
