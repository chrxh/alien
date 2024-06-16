#include "TimelineLiveStatistics.h"

#include <cmath>
#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/StatisticsConverterService.h"


void TimelineLiveStatistics::truncate()
{
    if (!dataPointCollectionHistory.empty() && dataPointCollectionHistory.back().time - dataPointCollectionHistory.front().time > (MaxLiveHistory + 1.0)) {
        dataPointCollectionHistory.erase(dataPointCollectionHistory.begin());
    }
}

void TimelineLiveStatistics::add(TimelineStatistics const& data, uint64_t timestep, double deltaTime)
{
    truncate();

    timepoint += deltaTime;

    auto newDataPoint = StatisticsConverterService::convert(data, timestep, timepoint, lastData, lastTimestep);
    dataPointCollectionHistory.emplace_back(newDataPoint);
    lastData = data;
    lastTimestep = timestep;
}
