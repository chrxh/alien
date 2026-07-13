#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/DataPointCollection.h>
#include <EngineInterface/OverallStatistics.h>

class StatisticsConverterService
{
    MAKE_SINGLETON(StatisticsConverterService);

public:
    DataPointCollection convert(OverallStatisticsEntry const& overallStatistics, uint64_t timestep, double time);
};
