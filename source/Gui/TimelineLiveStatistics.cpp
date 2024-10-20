#include "TimelineLiveStatistics.h"

#include <cmath>
#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/StatisticsConverterService.h"

std::vector<DataPointCollection> const& TimelineLiveStatistics::getDataPointCollectionHistory() const
{
    return _dataPointCollectionHistory;
}

void TimelineLiveStatistics::update(TimelineStatistics const& data, uint64_t timestep)
{
    truncate();

    auto timepoint = std::chrono::steady_clock::now();
    auto duration = _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;

    _timeSinceSimStart += toDouble(duration) / 1000;

    auto newDataPoint = StatisticsConverterService::get().convert(data, timestep, _timeSinceSimStart, _lastData, _lastTimestep);
    _dataPointCollectionHistory.emplace_back(newDataPoint);
    _lastData = data;
    _lastTimestep = timestep;
    _lastTimepoint = timepoint;
}

void TimelineLiveStatistics::truncate()
{
    if (!_dataPointCollectionHistory.empty() && _dataPointCollectionHistory.back().time - _dataPointCollectionHistory.front().time > (MaxLiveHistory + 1.0)) {
        _dataPointCollectionHistory.erase(_dataPointCollectionHistory.begin());
    }
}
