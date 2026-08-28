#pragma once

#include <cstdint>

#include <Base/Singleton.h>

#include <EngineInterface/ExtinctLineageAccumulator.h>
#include <EngineInterface/StatisticsEntry.h>

#include "LiveStatisticsHistory.h"

class LiveStatisticsService
{
    MAKE_SINGLETON(LiveStatisticsService);

public:
    static auto constexpr MaxLiveSteps = 100000;  // Max span retained for "Last time steps" mode

    // Rate metrics are averaged against the most recent sample at least this many time steps older
    static auto constexpr RateAveragingTimesteps = 1000.0;

    void addDataPoint(LiveStatisticsHistory& history, StatisticsEntry const& statisticsEntry, uint64_t timestep);

    // Discards the accumulated history, e.g. after a different simulation has been loaded
    void clear(LiveStatisticsHistory& history);

private:
    void truncate(LiveStatisticsHistory& history);

    ExtinctLineageAccumulator _extinctLineageAccumulator;
};
