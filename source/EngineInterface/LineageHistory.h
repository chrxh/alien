#pragma once

#include <mutex>
#include <vector>

#include "LineageStatistics.h"

struct LineageSample
{
    double time = 0;  // Could be a time step or real-time
    double systemClock = 0;
    std::vector<LineageStatisticsEntry> entries;  // Sorted by lineageId
};

using LineageHistoryData = std::vector<LineageSample>;

class LineageHistory
{
public:
    LineageHistoryData getCopiedData() const;

    std::mutex& getMutex() const;
    LineageHistoryData& getDataRef();
    LineageHistoryData const& getDataRef() const;

private:
    mutable std::mutex _mutex;
    LineageHistoryData _data;
};
