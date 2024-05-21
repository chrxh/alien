#include "StatisticsHistory.h"


StatisticsHistoryData StatisticsHistory::getCopiedData() const
{
    std::lock_guard lock(_mutex);
    auto copy = _data;
    return copy;
}

std::mutex& StatisticsHistory::getMutex() const
{
    return _mutex;
}

StatisticsHistoryData& StatisticsHistory::getDataRef()
{
    return _data;
}

StatisticsHistoryData const& StatisticsHistory::getDataRef() const
{
    return _data;
}
