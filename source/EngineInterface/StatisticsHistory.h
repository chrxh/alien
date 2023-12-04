#pragma once

#include <mutex>
#include <vector>

#include "DataPointCollection.h"
#include "Definitions.h"

using StatisticsHistoryData = std::vector<DataPointCollection>;

class StatisticsHistory
{
public:
    StatisticsHistoryData getCopiedData() const;

    std::mutex& getMutex() const;
    StatisticsHistoryData& getData();
    StatisticsHistoryData const& getData() const;

private:
    mutable std::mutex _mutex;
    StatisticsHistoryData _data;
};
