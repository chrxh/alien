#pragma once

#include <chrono>
#include <vector>
#include <optional>

#include "EngineInterface/Colors.h"
#include "EngineInterface/StatisticsRawData.h"
#include "EngineInterface/DataPointCollection.h"

class TimelineLiveStatistics
{
public:
    static auto constexpr MaxLiveHistory = 240.0f;  //in seconds

    std::vector<DataPointCollection> const& getDataPointCollectionHistory() const;
    void update(TimelineStatistics const& statistics, uint64_t timestep);

private:
    void truncate();

    double _timeSinceSimStart = 0;  //in seconds

    std::vector<DataPointCollection> _dataPointCollectionHistory;

    std::optional<uint64_t> _lastTimestep;
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
    std::optional<TimelineStatistics> _lastData;
};
