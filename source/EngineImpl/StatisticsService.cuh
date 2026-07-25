#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/ExtinctLineageAccumulator.h>
#include <EngineInterface/StatisticsEntry.h>
#include <EngineInterface/StatisticsHistory.h>

#include <EngineKernels/Definitions.cuh>

class StatisticsService
{
    MAKE_SINGLETON(StatisticsService);

public:
    void addDataPoint(StatisticsHistory& history, StatisticsEntry const& statisticsEntry, uint64_t timestep);
    void resetTime(StatisticsHistory& history, uint64_t timestep);
    void setStatisticsHistory(StatisticsHistory& history, StatisticsHistoryData const& newHistoryData, uint64_t timestep);

private:
    static auto constexpr DefaultTimeStepDelta = 10.0;

    template <typename DataPoint>
    struct TimelineState
    {
        double longtermTimestepDelta = DefaultTimeStepDelta;
        int numDataPoints = 0;
        std::optional<TimedSample<DataPoint>> accumulatedSample;
    };

    template <typename DataPoint>
    void updateTimeline(
        std::vector<TimedSample<DataPoint>>& samples,
        TimelineState<DataPoint>& state,
        TimedSample<DataPoint> const& rawSample,
        double time,
        bool forceSampling);

    template <typename DataPoint>
    void emitSample(std::vector<TimedSample<DataPoint>>& samples, TimelineState<DataPoint>& state, double time);

    template <typename DataPoint>
    void resetTimelineTime(std::vector<TimedSample<DataPoint>>& samples, TimelineState<DataPoint>& state, double time);

    TimelineState<std::unordered_map<uint32_t, ColorOverallDataPoint>> _colorOverallStates;
    std::unordered_map<uint32_t, TimelineState<LineageDataPoint>> _lineageStates;
    ExtinctLineageAccumulator _extinctLineageAccumulator;
    std::optional<uint64_t> _lastTimestep;
};
