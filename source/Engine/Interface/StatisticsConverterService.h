#include <optional>

#include <Base/Interface/Singleton.h>

#include <Data/Interface/DataPointCollection.h>

#include <Engine/Interface/StatisticsEntry.h>

class StatisticsConverterService
{
    MAKE_SINGLETON(StatisticsConverterService);

public:
    DataPointCollection convert(StatisticsEntry const& statisticsEntry, uint64_t timestep);
};
