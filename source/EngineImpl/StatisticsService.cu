
#include "StatisticsService.cuh"

#include <algorithm>
#include <chrono>

#include <EngineInterface/StatisticsConverterService.h>

#include <EngineKernels/Base.cuh>

namespace
{
    auto constexpr MaxSamples = 1000;
}

void StatisticsService::addDataPoint(StatisticsHistory& history, OverallStatisticsEntry const& overallStatistics, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();

    if (!historyData.empty() && historyData.back().time > toDouble(timestep) + NEAR_ZERO) {
        historyData.clear();
    }

    if (!_lastTimestep || historyData.empty() || toDouble(timestep) - historyData.back().time > _longtermTimestepDelta / 100 * (_numDataPoints + 1)) {
        auto newDataPoint = [&] {
            if (!_lastTimestep && !historyData.empty()) {

                //reuse last entry if no statistics is available
                auto result = historyData.back();
                result.time = toDouble(timestep);
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

void StatisticsService::addSample(LineageHistory& history, LineageStatistics const& lineageStatistics, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();

    if (!historyData.empty() && historyData.back().time > toDouble(timestep) + NEAR_ZERO) {
        historyData.clear();
        _lineageTimestepDelta = DefaultTimeStepDelta;
    }

    if (!historyData.empty() && toDouble(timestep) - historyData.back().time < _lineageTimestepDelta) {
        return;
    }

    LineageSample sample;
    sample.time = toDouble(timestep);
    auto now = std::chrono::system_clock::now();
    auto unixEpoch = std::chrono::time_point<std::chrono::system_clock>();
    sample.systemClock = toDouble(std::chrono::duration_cast<std::chrono::seconds>(now - unixEpoch).count());
    sample.entries = lineageStatistics.entries;
    std::sort(sample.entries.begin(), sample.entries.end(), [](auto const& lhs, auto const& rhs) { return lhs.lineageId < rhs.lineageId; });
    historyData.emplace_back(std::move(sample));

    if (historyData.size() > MaxSamples) {
        LineageHistoryData newData;
        newData.reserve(historyData.size() / 2 + 1);
        for (size_t i = 0; i < (historyData.size() - 1) / 2; ++i) {
            newData.emplace_back(mergeSamples(historyData.at(i * 2), historyData.at(i * 2 + 1)));
        }
        newData.emplace_back(historyData.back());
        historyData.swap(newData);

        _lineageTimestepDelta *= 2.0;
    }
}

void StatisticsService::reset()
{
    _lineageTimestepDelta = DefaultTimeStepDelta;
}

LineageSample StatisticsService::mergeSamples(LineageSample const& earlierSample, LineageSample const& laterSample)
{
    LineageSample result;
    result.time = earlierSample.time;
    result.systemClock = (earlierSample.systemClock + laterSample.systemClock) / 2;
    result.entries.reserve(earlierSample.entries.size() + laterSample.entries.size());

    auto mergeEntries = [](LineageStatisticsEntry const& lhs, LineageStatisticsEntry const& rhs) {
        LineageStatisticsEntry result;
        result.lineageId = lhs.lineageId;
        result.colorBitset = lhs.colorBitset | rhs.colorBitset;
        result.numCreatures = (lhs.numCreatures + rhs.numCreatures) / 2;
        result.numGenomes = (lhs.numGenomes + rhs.numGenomes) / 2;
        result.sumCreatureCells = (lhs.sumCreatureCells + rhs.sumCreatureCells) / 2;
        result.sumCreatureGenerations = (lhs.sumCreatureGenerations + rhs.sumCreatureGenerations) / 2;
        result.sumGenomeNodes = (lhs.sumGenomeNodes + rhs.sumGenomeNodes) / 2;
        result.sumMutationRates = (lhs.sumMutationRates + rhs.sumMutationRates) / 2;
        result.sumCreatureEnergy = (lhs.sumCreatureEnergy + rhs.sumCreatureEnergy) / 2;
        result.numCreatedCreatures = std::max(lhs.numCreatedCreatures, rhs.numCreatedCreatures);
        result.totalMutations = std::max(lhs.totalMutations, rhs.totalMutations);
        return result;
    };

    auto earlierIter = earlierSample.entries.begin();
    auto laterIter = laterSample.entries.begin();
    while (earlierIter != earlierSample.entries.end() || laterIter != laterSample.entries.end()) {
        if (laterIter == laterSample.entries.end() || (earlierIter != earlierSample.entries.end() && earlierIter->lineageId < laterIter->lineageId)) {
            result.entries.emplace_back(*earlierIter);
            ++earlierIter;
        } else if (earlierIter == earlierSample.entries.end() || laterIter->lineageId < earlierIter->lineageId) {
            result.entries.emplace_back(*laterIter);
            ++laterIter;
        } else {
            result.entries.emplace_back(mergeEntries(*earlierIter, *laterIter));
            ++earlierIter;
            ++laterIter;
        }
    }
    return result;
}
