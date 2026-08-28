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

    auto newDataPoint = StatisticsConverterService::get().convert(statisticsEntry, timestep);
    _extinctLineageAccumulator.addExtinctLineageValues(newDataPoint);
    history.getDataRef().emplace_back(newDataPoint);
}

void LiveStatisticsService::clear(LiveStatisticsHistory& history)
{
    history.getDataRef().clear();
    _extinctLineageAccumulator.reset();
}

void LiveStatisticsService::truncate(LiveStatisticsHistory& history)
{
    // Keep the samples of the "Last time steps" window, with a hard count cap to bound memory if the
    // simulation stalls (timestep barely advancing over real time)
    auto& dataPoints = history.getDataRef();
    if (dataPoints.size() <= 1) {
        return;
    }
    auto const& back = dataPoints.back();

    size_t windowStartIndex = 0;  // Oldest sample still needed by the window
    while (windowStartIndex + 1 < dataPoints.size() && back.timestep - dataPoints.at(windowStartIndex).timestep > MaxLiveSteps) {
        ++windowStartIndex;
    }

    // Also keep a rate reference for the oldest sample of the window; otherwise the rate plots would start
    // with a gap once the window is set to its maximum span
    auto firstIndex = windowStartIndex;
    while (firstIndex > 0 && dataPoints.at(firstIndex).timestep + RateAveragingTimesteps > dataPoints.at(windowStartIndex).timestep) {
        --firstIndex;
    }
    if (toInt(dataPoints.size()) > MaxSampleCount) {
        firstIndex = std::max(firstIndex, dataPoints.size() - MaxSampleCount);
    }
    dataPoints.erase(dataPoints.begin(), dataPoints.begin() + firstIndex);
}
