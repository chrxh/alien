#pragma once

#include <chrono>

#include "EngineInterface/RawStatisticsData.h"
#include "Definitions.h"

class TableLiveStatistics
{
public:
    bool isDataAvailable() const;

    uint64_t getCreatedCells() const;
    uint64_t getCreatedCellsPerSecond() const;

    void update(TimelineStatistics const& data);

private:
    std::optional<TimelineStatistics> _lastData;
    std::optional<TimelineStatistics> _lastSecondData;
    std::optional<std::chrono::steady_clock::time_point> _lastSecondTimepoint;
};
