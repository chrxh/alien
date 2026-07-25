#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/ExtinctLineageAccumulator.h>
#include <EngineInterface/StatisticsEntry.h>

#include "LiveStatisticsHistory.h"

class LiveStatisticsService
{
    MAKE_SINGLETON(LiveStatisticsService);

public:
    static auto constexpr MaxLiveHistory = 240.0f;  //in seconds
    static auto constexpr MaxLiveSteps = 100000;    //max span retained for "Last time steps" mode

    void addDataPoint(LiveStatisticsHistory& history, StatisticsEntry const& statisticsEntry, uint64_t timestep);

    //discards the accumulated history, e.g. after a different simulation has been loaded
    void clear(LiveStatisticsHistory& history);

private:
    void truncate(LiveStatisticsHistory& history);

    ExtinctLineageAccumulator _extinctLineageAccumulator;
    double _timeSinceSimStart = 0;  //in seconds
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
};
