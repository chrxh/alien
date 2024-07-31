#include "TableLiveStatistics.h"

bool TableLiveStatistics::isDataAvailable() const
{
    return _lastData.has_value();
}

namespace
{
    uint64_t sum(ColorVector<uint64_t> const& valueByColor)
    {
        uint64_t result = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result += valueByColor[i];
        }
        return result;
    }
}

uint64_t TableLiveStatistics::getCreatedCells() const
{
    return sum(_lastData->accumulated.numCreatedCells);
}

void TableLiveStatistics::update(TimelineStatistics const& data)
{
    _lastData = data;
    //auto timepoint = std::chrono::steady_clock::now();
    //auto duration =
    //    _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;
}
