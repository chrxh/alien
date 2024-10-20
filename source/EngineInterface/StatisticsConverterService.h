#include <optional>

#include "Base/Singleton.h"
#include "EngineInterface/DataPointCollection.h"

class StatisticsConverterService
{
    MAKE_SINGLETON(StatisticsConverterService);

public:
    DataPointCollection convert(
        TimelineStatistics const& newData,
        uint64_t timestep,
        double time,
        std::optional<TimelineStatistics> const& lastData,
        std::optional<uint64_t> lastTimestep);
};
