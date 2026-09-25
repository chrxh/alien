#include <optional>

#include <Base/Singleton.h>

#include <Data/DataPointCollection.h>

#include <EngineInterface/StatisticsEntry.h>

class StatisticsConverterService
{
    MAKE_SINGLETON(StatisticsConverterService);

public:
    DataPointCollection convert(StatisticsEntry const& statisticsEntry, uint64_t timestep);
};
