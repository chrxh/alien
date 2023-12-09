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

StatisticsHistoryData& StatisticsHistory::getDataRef()
{
    return _data;
}

StatisticsHistoryData const& StatisticsHistory::getDataRef() const
{
    return _data;
}
