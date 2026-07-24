
#include "StatisticsService.cuh"

#include <algorithm>
#include <cmath>

#include <EngineInterface/StatisticsConverterService.h>

#include <EngineKernels/Base.cuh>

namespace
{
    auto constexpr MaxSamples = 1000;
}

void StatisticsService::addDataPoint(StatisticsHistory& history, StatisticsEntry const& statisticsEntry, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();

    auto timestepAsDouble = toDouble(timestep);
    if (!historyData.colors.empty() && historyData.colors.back().timestep > timestepAsDouble + NEAR_ZERO) {
        historyData = StatisticsHistoryData();
        _colorOverallStates = TimelineState<std::unordered_map<uint32_t, ColorOverallDataPoint>>();
        _lineageStates.clear();
    }

    auto firstSampling = !_lastTimestep.has_value();
    if (firstSampling && !historyData.colors.empty()) {

        // Reuse last entries if no statistics is available
        auto overallBackTime = historyData.colors.back().timestep;
        auto overallSample = historyData.colors.back();
        overallSample.timestep = timestepAsDouble;
        updateTimeline(historyData.colors, _colorOverallStates, overallSample, timestepAsDouble, true);

        // Only lineages that were still sampled at the end of the history are continued; extinct ones keep their last entry
        for (auto& [lineageId, samples] : historyData.lineages) {
            if (samples.empty()) {
                continue;
            }
            auto& state = _lineageStates[lineageId];
            if (overallBackTime - samples.back().timestep > state.longtermTimestepDelta * 2) {
                continue;
            }
            auto lineageSample = samples.back();
            lineageSample.timestep = timestepAsDouble;
            updateTimeline(samples, state, lineageSample, timestepAsDouble, true);
        }
    } else {
        auto dataPoints = StatisticsConverterService::get().convert(statisticsEntry, timestep, timestepAsDouble);

        ColorSamples overallSample;
        overallSample.timestep = dataPoints.timestep;
        overallSample.systemClock = dataPoints.systemClock;
        overallSample.data = dataPoints.overall;
        updateTimeline(historyData.colors, _colorOverallStates, overallSample, timestepAsDouble, firstSampling);

        for (auto const& [lineageId, lineageDataPoint] : dataPoints.lineages) {
            LineageSample lineageSample;
            lineageSample.timestep = dataPoints.timestep;
            lineageSample.systemClock = dataPoints.systemClock;
            lineageSample.data = lineageDataPoint;
            updateTimeline(historyData.lineages[lineageId], _lineageStates[lineageId], lineageSample, timestepAsDouble, firstSampling);
        }

        // Drop extinct lineages (no longer present in the current statistics) so their history is neither kept nor serialized
        for (auto stateIt = _lineageStates.begin(); stateIt != _lineageStates.end();) {
            if (!dataPoints.lineages.contains(stateIt->first)) {
                historyData.lineages.erase(stateIt->first);
                stateIt = _lineageStates.erase(stateIt);
            } else {
                ++stateIt;
            }
        }
    }

    _lastTimestep = timestep;
}

void StatisticsService::resetTime(StatisticsHistory& history, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();
    auto timestepAsDouble = toDouble(timestep);

    resetTimelineTime(historyData.colors, _colorOverallStates, timestepAsDouble);

    for (auto timelineIt = historyData.lineages.begin(); timelineIt != historyData.lineages.end();) {
        auto& samples = timelineIt->second;
        if (auto stateIt = _lineageStates.find(timelineIt->first); stateIt != _lineageStates.end()) {
            resetTimelineTime(samples, stateIt->second, timestepAsDouble);
        } else {
            std::erase_if(samples, [&](LineageSample const& sample) { return sample.timestep >= timestepAsDouble; });
        }
        timelineIt = samples.empty() ? historyData.lineages.erase(timelineIt) : std::next(timelineIt);
    }
}

void StatisticsService::setStatisticsHistory(StatisticsHistory& history, StatisticsHistoryData const& newHistoryData, uint64_t timestep)
{
    _colorOverallStates = TimelineState<std::unordered_map<uint32_t, ColorOverallDataPoint>>();
    _lineageStates.clear();
    _lastTimestep.reset();

    if (!newHistoryData.colors.empty()) {
        _colorOverallStates.longtermTimestepDelta =
            std::max(DefaultTimeStepDelta, (toDouble(timestep) - newHistoryData.colors.front().timestep) / toDouble(newHistoryData.colors.size()));
    }
    for (auto const& [lineageId, samples] : newHistoryData.lineages) {
        if (!samples.empty()) {
            _lineageStates[lineageId].longtermTimestepDelta =
                std::max(DefaultTimeStepDelta, (samples.back().timestep - samples.front().timestep) / toDouble(samples.size()));
        }
    }

    std::lock_guard lock(history.getMutex());
    history.getDataRef() = newHistoryData;
}

template <typename DataPoint>
void StatisticsService::updateTimeline(
    std::vector<TimedSample<DataPoint>>& samples,
    TimelineState<DataPoint>& state,
    TimedSample<DataPoint> const& rawSample,
    double time,
    bool forceSampling)
{
    if (forceSampling || samples.empty() || time - samples.back().timestep > state.longtermTimestepDelta / 100 * (state.numDataPoints + 1)) {
        state.accumulatedSample = state.accumulatedSample.has_value() ? *state.accumulatedSample + rawSample : rawSample;
        ++state.numDataPoints;
    }

    if (state.accumulatedSample.has_value() && (samples.empty() || time - samples.back().timestep > state.longtermTimestepDelta)) {
        emitSample(samples, state, time);
    }
}

template <typename DataPoint>
void StatisticsService::emitSample(std::vector<TimedSample<DataPoint>>& samples, TimelineState<DataPoint>& state, double time)
{
    auto newSample = *state.accumulatedSample / toDouble(state.numDataPoints);
    state.numDataPoints = 0;
    state.accumulatedSample.reset();

    // Remove last entry if timestep has not changed
    if (!samples.empty() && std::abs(samples.back().timestep - time) < NEAR_ZERO) {
        samples.pop_back();
    }
    samples.emplace_back(newSample);

    // Compress timeline after MaxSamples
    if (samples.size() > MaxSamples) {
        std::vector<TimedSample<DataPoint>> newSamples;
        newSamples.reserve(samples.size() / 2);
        for (size_t i = 0; i < (samples.size() - 1) / 2; ++i) {
            auto interpolatedSample = (samples.at(i * 2) + samples.at(i * 2 + 1)) / 2.0;
            interpolatedSample.timestep = samples.at(i * 2).timestep;
            newSamples.emplace_back(interpolatedSample);
        }
        newSamples.emplace_back(samples.back());
        samples.swap(newSamples);

        state.longtermTimestepDelta *= 2.0;
    }
}

template <typename DataPoint>
void StatisticsService::resetTimelineTime(std::vector<TimedSample<DataPoint>>& samples, TimelineState<DataPoint>& state, double time)
{
    if (!samples.empty() && samples.back().timestep > 0) {
        state.longtermTimestepDelta = std::max(DefaultTimeStepDelta, state.longtermTimestepDelta * time / samples.back().timestep);
    } else {
        state.longtermTimestepDelta = DefaultTimeStepDelta;
    }
    std::erase_if(samples, [&](TimedSample<DataPoint> const& sample) { return sample.timestep >= time; });
    state.accumulatedSample.reset();
    state.numDataPoints = 0;
}
