#include "TableLiveStatistics.h"

bool TableLiveStatistics::isDataAvailable() const
{
    return _currentData.has_value();
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

float TableLiveStatistics::getCreatedCellsPerSecond() const
{
    if (!_lastDataTimepoint.has_value()) {
        return 0;
    }

    auto currentCreatedCells = sum(_currentData->accumulated.numCreatedCells);
    auto lastCreatedCells = sum(_lastData->accumulated.numCreatedCells);
    return (toFloat(currentCreatedCells) - toFloat(lastCreatedCells))
        / toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(*_currentDataTimepoint - *_lastDataTimepoint).count()) * 5000;
}

float TableLiveStatistics::getReplicatorsPerSecond() const
{
    return 0;
}

void TableLiveStatistics::update(TimelineStatistics const& data)
{
    auto timepoint = std::chrono::steady_clock::now();

    if (!_currentDataTimepoint.has_value()) {
        _currentData = data;
        _currentDataTimepoint = timepoint;
        return;
    }

    auto duration = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_currentDataTimepoint).count());
    if (duration > 5000) {
        _lastData = _currentData;
        _lastDataTimepoint = _currentDataTimepoint;

        _currentData = data;
        _currentDataTimepoint = timepoint;
    }
}
