#include "TimelineLiveStatistics.h"

#include <cmath>

#include <imgui.h>

#include <Base/Definitions.h>

#include <EngineInterface/StatisticsConverterService.h>

namespace
{
    auto constexpr MaxSampleCount = 20000;  //hard cap to bound memory when the simulation stalls
}

std::vector<DataPointCollection> const& TimelineLiveStatistics::getDataPointCollectionHistory() const
{
    return _dataPointCollectionHistory;
}

void TimelineLiveStatistics::clear()
{
    _dataPointCollectionHistory.clear();
}

void TimelineLiveStatistics::update(StatisticsEntry const& overallStatistics, uint64_t timestep)
{
    truncate();

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
    //keep enough history to cover both the real-time window (time-based) and the "Last X steps" window
    //(step-based); a sample is only dropped once it is no longer needed by either, with a hard count cap
    //to bound memory if the simulation stalls (timestep barely advancing over real time)
    while (_dataPointCollectionHistory.size() > 1) {
        auto const& front = _dataPointCollectionHistory.front();
        auto const& back = _dataPointCollectionHistory.back();
        auto exceedsSampleCount = toInt(_dataPointCollectionHistory.size()) > MaxSampleCount;
        auto exceedsTimeAndStepWindow = back.time - front.time > (MaxLiveHistory + 1.0) && back.timestep - front.timestep > MaxLiveSteps;
        if (!exceedsSampleCount && !exceedsTimeAndStepWindow) {
            break;
        }
        _dataPointCollectionHistory.erase(_dataPointCollectionHistory.begin());
    }
}
