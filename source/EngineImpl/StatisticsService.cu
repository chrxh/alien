
#include "StatisticsService.cuh"

#include <EngineInterface/StatisticsConverterService.h>

#include <EngineKernels/Base.cuh>

namespace
{
    auto constexpr MaxSamples = 1000;
}

void StatisticsService::addDataPoint(StatisticsHistory& history, StatisticsEntry const& overallStatistics, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();

    if (!historyData.empty() && historyData.back().time > toDouble(timestep) + NEAR_ZERO) {
        historyData.clear();
        _longtermTimestepDelta = DefaultTimeStepDelta;
    }

    if (!_lastTimestep || historyData.empty() || toDouble(timestep) - historyData.back().time > _longtermTimestepDelta / 100 * (_numDataPoints + 1)) {
        auto newDataPoint = [&]() -> DataPointCollection {
            if (!_lastTimestep && !historyData.empty()) {

                // Reuse last entry if no statistics is available
                auto result = historyData.back();
                result.time = toDouble(timestep);
                result.timestep = toDouble(timestep);
                return result;
            } else {
                return StatisticsConverterService::get().convert(overallStatistics, timestep, toDouble(timestep));
            }
        }();

        _lastTimestep = timestep;
        _accumulatedDataPoint = _accumulatedDataPoint.has_value() ? *_accumulatedDataPoint + newDataPoint : newDataPoint;
        ++_numDataPoints;
    }

    if (_accumulatedDataPoint.has_value() && (historyData.empty() || toDouble(timestep) - historyData.back().time > _longtermTimestepDelta)) {
        auto newDataPoint = *_accumulatedDataPoint / _numDataPoints;
        _numDataPoints = 0;
        _accumulatedDataPoint.reset();

        // Remove last entry if timestep has not changed
        if (!historyData.empty() && abs(historyData.back().time - toDouble(timestep)) < NEAR_ZERO) {
            historyData.pop_back();
        }
        historyData.emplace_back(newDataPoint);

        // Compress history after MaxSamples
        if (historyData.size() > MaxSamples) {
            StatisticsHistoryData newData;
            newData.reserve(historyData.size() / 2);
            for (size_t i = 0; i < (historyData.size() - 1) / 2; ++i) {
                auto interpolatedDataPoint = (historyData.at(i * 2) + historyData.at(i * 2 + 1)) / 2.0;
                interpolatedDataPoint.time = historyData.at(i * 2).time;
                interpolatedDataPoint.timestep = historyData.at(i * 2).timestep;
                newData.emplace_back(interpolatedDataPoint);
            }
            newData.emplace_back(historyData.back());
            historyData.swap(newData);

            _longtermTimestepDelta *= 2.0;
        }
    }
}

void StatisticsService::resetTime(StatisticsHistory& history, uint64_t timestep)
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

    StatisticsHistoryData newData;
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

void StatisticsService::rewriteHistory(StatisticsHistory& history, StatisticsHistoryData const& newHistoryData, uint64_t timestep)
{
    _accumulatedDataPoint.reset();
    _numDataPoints = 0;
    _lastTimestep.reset();
    if (!newHistoryData.empty()) {
        _longtermTimestepDelta = max(DefaultTimeStepDelta, (timestep - newHistoryData.front().time) / toDouble(newHistoryData.size()));
    } else {
        _longtermTimestepDelta = DefaultTimeStepDelta;
    }

    std::lock_guard lock(history.getMutex());
    history.getDataRef() = newHistoryData;
}
