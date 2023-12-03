#include <optional>

#include "EngineInterface/DataPointCollection.h"

class StatisticsConverterService
{
public:
    static DataPointCollection
    convert(TimelineStatistics const& newData, uint64_t timestep, std::optional<TimelineStatistics> const& lastData, std::optional<uint64_t> lastTimestep);
};
