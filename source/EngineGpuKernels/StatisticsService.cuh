#include <optional>

#include "Base/Singleton.h"
#include "EngineInterface/StatisticsHistory.h"

#include "Definitions.cuh"

class StatisticsService
{
    MAKE_SINGLETON(StatisticsService);

public:
    void addDataPoint(StatisticsHistory& history, TimelineStatistics const& newRawStatistics, uint64_t timestep);
    void resetTime(StatisticsHistory& history, uint64_t timestep);
    void rewriteHistory(StatisticsHistory& history, StatisticsHistoryData const& newHistoryData, uint64_t timestep);

private:
    static auto constexpr DefaultTimeStepDelta = 10.0;

    double _longtermTimestepDelta = DefaultTimeStepDelta;

    int _numDataPoints = 0;
    std::optional<DataPointCollection> _accumulatedDataPoint;

    std::optional<TimelineStatistics> _lastRawStatistics;
    std::optional<uint64_t> _lastTimestep;
};
