#include "LiveStatisticsService.h"

#include <Base/Definitions.h>

#include <EngineInterface/StatisticsConverterService.h>

namespace
{
    auto constexpr MaxSampleCount = 20000;  //hard cap to bound memory when the simulation stalls
}

void LiveStatisticsService::addDataPoint(LiveStatisticsHistory& history, StatisticsEntry const& statisticsEntry, uint64_t timestep)
{
    truncate(history);

    auto timepoint = std::chrono::steady_clock::now();
    auto duration =
        _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;

    _timeSinceSimStart += toDouble(duration) / 1000;

    auto newDataPoint = StatisticsConverterService::get().convert(statisticsEntry, timestep, _timeSinceSimStart);
    history.getDataRef().emplace_back(newDataPoint);
    _lastTimepoint = timepoint;
}

void LiveStatisticsService::clear(LiveStatisticsHistory& history)
{
    history.getDataRef().clear();
}

void LiveStatisticsService::truncate(LiveStatisticsHistory& history)
{
    //keep enough history to cover both the real-time window (time-based) and the "Last time steps" window
    //(step-based); a sample is only dropped once it is no longer needed by either, with a hard count cap
    //to bound memory if the simulation stalls (timestep barely advancing over real time)
    auto& dataPoints = history.getDataRef();
    while (dataPoints.size() > 1) {
        auto const& front = dataPoints.front();
        auto const& back = dataPoints.back();
        auto exceedsSampleCount = toInt(dataPoints.size()) > MaxSampleCount;
        auto exceedsTimeAndStepWindow = back.time - front.time > (MaxLiveHistory + 1.0) && back.timestep - front.timestep > MaxLiveSteps;
        if (!exceedsSampleCount && !exceedsTimeAndStepWindow) {
            break;
        }
        dataPoints.erase(dataPoints.begin());
    }
}
