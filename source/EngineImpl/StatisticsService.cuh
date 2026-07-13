#pragma once

#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/LineageHistory.h>
#include <EngineInterface/LineageStatistics.h>
#include <EngineInterface/OverallStatistics.h>
#include <EngineInterface/StatisticsHistory.h>

#include <EngineKernels/Definitions.cuh>

class StatisticsService
{
    MAKE_SINGLETON(StatisticsService);

public:
    void addDataPoint(StatisticsHistory& history, OverallStatisticsEntry const& overallStatistics, uint64_t timestep);
    void resetTime(StatisticsHistory& history, uint64_t timestep);
    void rewriteHistory(StatisticsHistory& history, StatisticsHistoryData const& newHistoryData, uint64_t timestep);

    void addSample(LineageHistory& history, LineageStatistics const& lineageStatistics, uint64_t timestep);
    void reset();

private:
    LineageSample mergeSamples(LineageSample const& earlierSample, LineageSample const& laterSample);

    static auto constexpr DefaultTimeStepDelta = 10.0;

    double _longtermTimestepDelta = DefaultTimeStepDelta;
    int _numDataPoints = 0;
    std::optional<DataPointCollection> _accumulatedDataPoint;
    std::optional<uint64_t> _lastTimestep;

    double _lineageTimestepDelta = DefaultTimeStepDelta;
};
