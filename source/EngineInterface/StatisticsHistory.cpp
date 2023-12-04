#include "StatisticsHistory.h"


StatisticsHistoryData StatisticsHistory::getCopiedData() const
{
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
