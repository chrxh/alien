#include <optional>

#include "EngineInterface/StatisticsHistory.h"
#include "Definitions.cuh"

class _StatisticsService
{
public:
    void addDataPoint(StatisticsHistory& history, TimelineStatistics const& statisticsToAdd, uint64_t timestep);

private:
    double longtermTimestepDelta = 10.0;

    std::optional<TimelineStatistics> lastData;
    std::optional<uint64_t> lastTimestep;
};
