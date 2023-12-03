#include "StatisticsHistory.h"


std::vector<DataPointCollection>& StatisticsHistory::getData()
{
    return _data;
}

std::vector<DataPointCollection> const& StatisticsHistory::getData() const
{
    return _data;
}
