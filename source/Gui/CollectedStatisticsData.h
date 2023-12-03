#pragma once

#include <vector>
#include <optional>

#include "EngineInterface/Colors.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/DataPointCollection.h"

struct TimelineLiveStatistics
{
    static double constexpr MaxLiveHistory = 240.0;  //in seconds

    double timepoint = 0;  //in seconds
    float history = 10.0f;   //in seconds

    std::vector<DataPointCollection> dataPointCollectionHistory;
    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;

    void truncate();
    void add(TimelineStatistics const& statistics, uint64_t timestep);
};

struct TimelineLongtermStatistics
{
    double longtermTimestepDelta = 10.0;

    std::vector<DataPointCollection> dataPointCollectionHistory;
    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;

    void add(TimelineStatistics const& statistics, uint64_t timestep);
};
