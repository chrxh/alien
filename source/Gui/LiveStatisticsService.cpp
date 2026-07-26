#include "LiveStatisticsService.h"

#include <algorithm>

#include <Base/Definitions.h>

#include <EngineInterface/StatisticsConverterService.h>

namespace
{
    auto constexpr MaxSampleCount = 20000;  // Hard cap to bound memory when the simulation stalls
}

void LiveStatisticsService::addDataPoint(LiveStatisticsHistory& history, StatisticsEntry const& statisticsEntry, uint64_t timestep)
{
    truncate(history);

    auto timepoint = std::chrono::steady_clock::now();
    auto duration =
        _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;

    _timeSinceSimStart += toDouble(duration) / 1000;

    auto newDataPoint = StatisticsConverterService::get().convert(statisticsEntry, timestep, _timeSinceSimStart);
    _extinctLineageAccumulator.addExtinctLineageValues(newDataPoint);
    history.getDataRef().emplace_back(newDataPoint);
    _lastTimepoint = timepoint;
}

void LiveStatisticsService::clear(LiveStatisticsHistory& history)
{
    history.getDataRef().clear();
    _extinctLineageAccumulator.reset();
}

void LiveStatisticsService::truncate(LiveStatisticsHistory& history)
{
    // Keep enough history to cover both the real-time window (time-based) and the "Last time steps" window
    // (step-based); a sample is only dropped once it is no longer needed by either, with a hard count cap
    // to bound memory if the simulation stalls (timestep barely advancing over real time)
    auto& dataPoints = history.getDataRef();
    if (dataPoints.size() <= 1) {
        return;
    }
    auto const& back = dataPoints.back();

    size_t windowStartIndex = 0;  // Oldest sample still needed by one of the two windows
    while (windowStartIndex + 1 < dataPoints.size()) {
        auto const& candidate = dataPoints.at(windowStartIndex);
        if (back.time - candidate.time <= MaxLiveHistory + 1.0 || back.timestep - candidate.timestep <= MaxLiveSteps) {
            break;
        }
        ++windowStartIndex;
    }

    // Also keep a rate reference for the oldest sample of the windows; otherwise the rate plots would start
    // with a gap once the windows are set to their maximum span
    auto firstIndex = windowStartIndex;
    while (firstIndex > 0 && dataPoints.at(firstIndex).timestep + RateAveragingTimesteps > dataPoints.at(windowStartIndex).timestep) {
        --firstIndex;
    }
    if (toInt(dataPoints.size()) > MaxSampleCount) {
        firstIndex = std::max(firstIndex, dataPoints.size() - MaxSampleCount);
    }
    dataPoints.erase(dataPoints.begin(), dataPoints.begin() + firstIndex);
}
