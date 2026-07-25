#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/DataPointCollection.h>
#include <EngineInterface/StatisticsEntry.h>

class StatisticsConverterService
{
    MAKE_SINGLETON(StatisticsConverterService);

public:
    DataPointCollection convert(StatisticsEntry const& statisticsEntry, uint64_t timestep, double time);
};
