#include "TimelineLiveStatistics.h"

#include <cmath>

#include <imgui.h>

#include <Base/Definitions.h>

#include <EngineInterface/StatisticsConverterService.h>

std::vector<DataPointCollection> const& TimelineLiveStatistics::getDataPointCollectionHistory() const
{
    return _dataPointCollectionHistory;
}

bool TimelineLiveStatistics::wasReset() const
{
    return _wasReset;
}

void TimelineLiveStatistics::update(StatisticsEntry const& overallStatistics, uint64_t timestep)
{
    _wasReset = _lastTimestep.has_value() && timestep < *_lastTimestep;
    if (_wasReset) {
        _dataPointCollectionHistory.clear();
    } else {
        truncate();
    }

    auto timepoint = std::chrono::steady_clock::now();
    auto duration =
        _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;

    _timeSinceSimStart += toDouble(duration) / 1000;

    auto newDataPoint = StatisticsConverterService::get().convert(overallStatistics, timestep, _timeSinceSimStart);
    _dataPointCollectionHistory.emplace_back(newDataPoint);
    _lastTimestep = timestep;
    _lastTimepoint = timepoint;
}

void TimelineLiveStatistics::truncate()
{
    if (!_dataPointCollectionHistory.empty() && _dataPointCollectionHistory.back().time - _dataPointCollectionHistory.front().time > (MaxLiveHistory + 1.0)) {
        _dataPointCollectionHistory.erase(_dataPointCollectionHistory.begin());
    }
}
