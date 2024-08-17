#include "TableLiveStatistics.h"

namespace
{
    auto constexpr TimeInterval = 5000; //in millisec

    uint64_t sum(ColorVector<uint64_t> const& valueByColor)
    {
        uint64_t result = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result += valueByColor[i];
        }
        return result;
    }
}

bool TableLiveStatistics::isDataAvailable() const
{
    return _currentData.has_value();
}

float TableLiveStatistics::getCreatedCellsPerSecond() const
{
    if (!_lastDataTimepoint.has_value()) {
        return 0;
    }
    return calcObjectsPerSecond(_lastData->accumulated.numCreatedCells, _currentData->accumulated.numCreatedCells);
}

float TableLiveStatistics::getCreatedReplicatorsPerSecond() const
{
    return calcObjectsPerSecond(_lastData->accumulated.numCreatedReplicators, _currentData->accumulated.numCreatedReplicators);
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
    if (duration > TimeInterval) {
        _lastData = _currentData;
        _lastDataTimepoint = _currentDataTimepoint;

        _currentData = data;
        _currentDataTimepoint = timepoint;
    }
}

float TableLiveStatistics::calcObjectsPerSecond(ColorVector<uint64_t> const& lastCount, ColorVector<uint64_t> const& currentCount) const
{
    auto currentCreatedObjects = sum(currentCount);
    auto lastCreatedObjects = sum(lastCount);
    return (toFloat(currentCreatedObjects) - toFloat(lastCreatedObjects))
        / toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(*_currentDataTimepoint - *_lastDataTimepoint).count()) * TimeInterval;
}
