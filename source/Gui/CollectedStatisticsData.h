#pragma once

#include <vector>

#include "EngineInterface/Colors.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/StatisticsData.h"
#include "EngineInterface/DataPointCollection.h"

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
