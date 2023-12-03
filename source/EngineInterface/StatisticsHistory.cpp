#include "StatisticsHistory.h"


std::mutex& StatisticsHistory::getMutex() const
{
    return _mutex;
}

std::vector<DataPointCollection>& StatisticsHistory::getData()
{
    return _data;
}

std::vector<DataPointCollection> const& StatisticsHistory::getData() const
{
    return _data;
}
