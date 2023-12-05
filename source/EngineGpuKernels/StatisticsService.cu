#include "StatisticsService.cuh"

#include "EngineInterface/StatisticsConverterService.h"

#include "Base.cuh"

namespace 
{
    auto constexpr MaxSamples = 1000;
}

void _StatisticsService::addDataPoint(StatisticsHistory& history, TimelineStatistics const& statisticsToAdd, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& data = history.getData();

    if (!data.empty() && data.back().time > toDouble(timestep) + NEAR_ZERO) {
        data.clear();
    }

    if (!_lastData || data.empty() || toDouble(timestep) - data.back().time > _longtermTimestepDelta) {

        auto newDataPoint = StatisticsConverterService::convert(statisticsToAdd, timestep, _lastData, _lastTimestep);
        newDataPoint.time = toDouble(timestep);

        if (!data.empty() && abs(data.back().time - toDouble(timestep)) < NEAR_ZERO) {
            data.pop_back();
        }
        data.emplace_back(newDataPoint);

        _lastData = statisticsToAdd;
        _lastTimestep = timestep;

        if (data.size() > MaxSamples) {
            std::vector<DataPointCollection> newData;
            newData.reserve(data.size() / 2);
            for (size_t i = 0; i < (data.size() - 1) / 2; ++i) {
                DataPointCollection newDataPoint = (data.at(i * 2) + data.at(i * 2 + 1)) / 2.0;
                newDataPoint.time = data.at(i * 2).time;
                newData.emplace_back(newDataPoint);
            }
            newData.emplace_back(data.back());
            data.swap(newData);

            _longtermTimestepDelta *= 2;
        }
    }
}

void _StatisticsService::resetTime(StatisticsHistory& history, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& data = history.getData();

    auto prevTimestep = data.back().time;
    if (!data.empty() && prevTimestep > 0) {
        _longtermTimestepDelta *= toDouble(timestep) / prevTimestep;
        if (_longtermTimestepDelta < DefaultTimeStepDelta) {
            _longtermTimestepDelta = DefaultTimeStepDelta;
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

void _StatisticsService::rewriteHistory(StatisticsHistory& history, StatisticsHistoryData const& data, uint64_t timestep)
{
    _lastData.reset();
    _lastTimestep.reset();
    if (data.empty()) {
        _longtermTimestepDelta = max(DefaultTimeStepDelta, (data.back().time - data.front().time) / (toDouble(data.size())));
    } else {
        _longtermTimestepDelta = DefaultTimeStepDelta;
    }

    std::lock_guard lock(history.getMutex());
    history.getData() = data;
}
