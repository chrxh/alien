#pragma once

#include <chrono>

#include "EngineInterface/RawStatisticsData.h"
#include "Definitions.h"

class TableLiveStatistics
{
public:
    bool isDataAvailable() const;

    float getCreatedCellsPerSecond() const;
    float getReplicatorsPerSecond() const;

    void update(TimelineStatistics const& data);

private:
    std::optional<TimelineStatistics> _currentData;
    std::optional<std::chrono::steady_clock::time_point> _currentDataTimepoint;

    std::optional<TimelineStatistics> _lastData;
    std::optional<std::chrono::steady_clock::time_point> _lastDataTimepoint;
};
