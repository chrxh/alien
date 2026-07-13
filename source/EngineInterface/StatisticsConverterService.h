#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/DataPointCollection.h>
#include <EngineInterface/OverallStatistics.h>

class StatisticsConverterService
{
    MAKE_SINGLETON(StatisticsConverterService);

public:
    DataPointCollection convert(
        TimelineStatistics const& newData,
        OverallStatisticsEntry const& overallStatistics,
        uint64_t timestep,
        double time,
        std::optional<TimelineStatistics> const& lastData,
        std::optional<uint64_t> lastTimestep);
};
