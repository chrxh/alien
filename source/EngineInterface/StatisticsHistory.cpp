#include "StatisticsHistory.h"


StatisticsHistoryData StatisticsHistory::getCopiedData() const
{
    std::lock_guard lock(_mutex);
    return _data;
}

std::mutex& StatisticsHistory::getMutex() const
{
    return _mutex;
}

StatisticsHistoryData& StatisticsHistory::getData()
{
    return _data;
}

StatisticsHistoryData const& StatisticsHistory::getData() const
{
    return _data;
}
