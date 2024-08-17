#include "HistogramLiveStatistics.h"

bool HistogramLiveStatistics::isDataAvailable() const
{
    return _lastHistogramData.has_value();
}

std::optional<HistogramData> const& HistogramLiveStatistics::getData() const
{
    return _lastHistogramData;
}

void HistogramLiveStatistics::update(HistogramData const& data)
{
    _lastHistogramData = data;
}
