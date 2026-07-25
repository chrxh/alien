#pragma once

#include <chrono>
#include <optional>
#include <vector>

#include <EngineInterface/DataPointCollection.h>
#include <EngineInterface/StatisticsEntry.h>

class TimelineLiveStatistics
{
public:
    static auto constexpr MaxLiveHistory = 240.0f;  //in seconds
    static auto constexpr MaxLiveSteps = 100000;    //max span retained for "Last time steps" mode

    std::vector<DataPointCollection> const& getDataPointCollectionHistory() const;
    void update(StatisticsEntry const& statisticsEntry, uint64_t timestep);

    //discards the accumulated history, e.g. after a different simulation has been loaded
    void clear();

private:
    void truncate();

    double _timeSinceSimStart = 0;  //in seconds

    std::vector<DataPointCollection> _dataPointCollectionHistory;

    std::optional<uint64_t> _lastTimestep;
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
};
