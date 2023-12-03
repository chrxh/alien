#include "StatisticsService.cuh"

#include "EngineInterface/StatisticsConverterService.h"

#include "Base.cuh"

void _StatisticsService::addDataPoint(StatisticsHistory& history, TimelineStatistics const& statisticsToAdd, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& data = history.getData();
    if (!lastData || data.empty() || toDouble(timestep) - data.back().time > longtermTimestepDelta) {

        auto newDataPoint = StatisticsConverterService::convert(statisticsToAdd, timestep, lastData, lastTimestep);
        newDataPoint.time = toDouble(timestep);
        data.emplace_back(newDataPoint);
        lastData = statisticsToAdd;
        lastTimestep = timestep;

        if (data.size() > 1000) {
            std::vector<DataPointCollection> newData;
            newData.reserve(data.size() / 2);
            for (size_t i = 0; i < (data.size() - 1) / 2; ++i) {
                DataPointCollection newDataPoint = (data.at(i * 2) + data.at(i * 2 + 1)) / 2.0;
                newDataPoint.time = data.at(i * 2).time;
                newData.emplace_back(newDataPoint);
            }
            newData.emplace_back(data.back());
            data.swap(newData);

            longtermTimestepDelta *= 2;
        }
    }
}

void _StatisticsService::resetTime(StatisticsHistory& history, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& data = history.getData();

    auto prevTimestep = data.back().time;
    if (!data.empty() && prevTimestep > 0) {
        longtermTimestepDelta *= toDouble(timestep) / prevTimestep;
        if (longtermTimestepDelta < 10.0) {
            longtermTimestepDelta = 10.0;
        }
    }
    
    std::vector<DataPointCollection> newData;
    newData.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (data.at(i).time < timestep) {
            newData.emplace_back(data.at(i));
        }
    }
    data.swap(newData);
}
