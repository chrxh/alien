#include "StatisticsService.cuh"

#include "EngineInterface/StatisticsConverterService.h"

#include "Base.cuh"

namespace 
{
    auto constexpr MaxSamples = 1000;
}

void _StatisticsService::addDataPoint(StatisticsHistory& history, TimelineStatistics const& newRawStatistics, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();

    if (!historyData.empty() && historyData.back().time > toDouble(timestep) + NEAR_ZERO) {
        historyData.clear();
    }

    if (!_lastRawStatistics || historyData.empty() || toDouble(timestep) - historyData.back().time > _longtermTimestepDelta / 100 * (_numDataPoints + 1)) {
        auto newDataPoint = [&] {
            if (!_lastRawStatistics && !historyData.empty()) {

                //reuse last entry if no raw statistics is available
                auto result = historyData.back();
                result.time = toDouble(timestep);
                return result;
            } else {
                return StatisticsConverterService::convert(newRawStatistics, timestep, toDouble(timestep), _lastRawStatistics, _lastTimestep);
            }
        }();

        _lastRawStatistics = newRawStatistics;
        _lastTimestep = timestep;
        _accumulatedDataPoint = _accumulatedDataPoint.has_value() ? *_accumulatedDataPoint + newDataPoint : newDataPoint;
        ++_numDataPoints;
    }

    if (_accumulatedDataPoint.has_value() && (historyData.empty() || toDouble(timestep) - historyData.back().time > _longtermTimestepDelta)) {
        auto newDataPoint = *_accumulatedDataPoint / _numDataPoints;
        _numDataPoints = 0;
        _accumulatedDataPoint.reset();

        //remove last entry if timestep has not changed
        if (!historyData.empty() && abs(historyData.back().time - toDouble(timestep)) < NEAR_ZERO) {
            historyData.pop_back();
        }
        historyData.emplace_back(newDataPoint);

        //compress history after MaxSamples
        if (historyData.size() > MaxSamples) {
            std::vector<DataPointCollection> newData;
            newData.reserve(historyData.size() / 2);
            for (size_t i = 0; i < (historyData.size() - 1) / 2; ++i) {
                DataPointCollection interpolatedDataPoint = (historyData.at(i * 2) + historyData.at(i * 2 + 1)) / 2.0;
                interpolatedDataPoint.time = historyData.at(i * 2).time;
                newData.emplace_back(interpolatedDataPoint);
            }
            newData.emplace_back(historyData.back());
            historyData.swap(newData);

            _longtermTimestepDelta *= 2.0;
        }
    }
}

void _StatisticsService::resetTime(StatisticsHistory& history, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& data = history.getDataRef();
    if (data.empty()) {
        return;
    }

    auto prevTimestep = data.back().time;
    if (!data.empty() && prevTimestep > 0) {
        _longtermTimestepDelta *= toDouble(timestep) / prevTimestep;
        if (_longtermTimestepDelta < DefaultTimeStepDelta) {
            _longtermTimestepDelta = DefaultTimeStepDelta;
        }
    } else {
        _longtermTimestepDelta = DefaultTimeStepDelta;
    }
    
    std::vector<DataPointCollection> newData;
    newData.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (data.at(i).time < toDouble(timestep)) {
            newData.emplace_back(data.at(i));
        }
    }
    data.swap(newData);
    _accumulatedDataPoint.reset();
    _numDataPoints = 0;
}

void _StatisticsService::rewriteHistory(StatisticsHistory& history, StatisticsHistoryData const& newHistoryData, uint64_t timestep)
{
    _accumulatedDataPoint.reset();
    _numDataPoints = 0;
    _lastRawStatistics.reset();
    _lastTimestep.reset();
    if (!newHistoryData.empty()) {
        _longtermTimestepDelta = max(DefaultTimeStepDelta, (timestep - newHistoryData.front().time) / toDouble(newHistoryData.size()));
    } else {
        _longtermTimestepDelta = DefaultTimeStepDelta;
    }

    std::lock_guard lock(history.getMutex());
    history.getDataRef() = newHistoryData;
}
