#include "LiveStatisticsHistory.h"

std::vector<DataPointCollection>& LiveStatisticsHistory::getDataRef()
{
    return _data;
}

std::vector<DataPointCollection> const& LiveStatisticsHistory::getDataRef() const
{
    return _data;
}
