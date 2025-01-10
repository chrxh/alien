#pragma once

#include <chrono>

#include "EngineInterface/StatisticsRawData.h"
#include "Definitions.h"

class TableLiveStatistics
{
public:
    bool isDataAvailable() const;

    float getCreatedCellsPerSecond() const;
    float getCreatedReplicatorsPerSecond() const;

    void update(TimelineStatistics const& data);

private:
    float calcObjectsPerSecond(ColorVector<uint64_t> const& lastCount, ColorVector<uint64_t> const& currentCount) const;

    std::optional<TimelineStatistics> _currentData;
    std::optional<std::chrono::steady_clock::time_point> _currentDataTimepoint;

    std::optional<TimelineStatistics> _lastData;
    std::optional<std::chrono::steady_clock::time_point> _lastDataTimepoint;
};
