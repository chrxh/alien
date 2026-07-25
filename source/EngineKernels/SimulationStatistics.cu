#include "SimulationStatistics.cuh"

#include <algorithm>

void SimulationStatistics::init()
{
    CudaMemoryManager::getInstance().acquireMemory<StatisticsControl>(1, _control);
    CudaMemoryManager::getInstance().acquireMemory<ObjectStatisticsEntry>(1, _objectStatisticsEntry);
    CudaMemoryManager::getInstance().acquireMemory<LineageStatisticsEntry>(LineageMapCapacity, _lineageStatisticsEntries);
    _lineageMap.init();
    for (auto& map : _accumulatorMaps) {
        map.init();
    }

    CHECK_FOR_DEVICE_ERRORS(cudaMemset(_control, 0, sizeof(StatisticsControl)));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(_objectStatisticsEntry, 0, sizeof(ObjectStatisticsEntry)));
}

void SimulationStatistics::free()
{
    CudaMemoryManager::getInstance().freeMemory(_control);
    CudaMemoryManager::getInstance().freeMemory(_objectStatisticsEntry);
    CudaMemoryManager::getInstance().freeMemory(_lineageStatisticsEntries);
    _lineageMap.free();
    for (auto& map : _accumulatorMaps) {
        map.free();
    }
}

StatisticsEntry SimulationStatistics::getStatisticsEntry() const
{
    StatisticsEntry result;
    CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result.objectStatistics, _objectStatisticsEntry, sizeof(ObjectStatisticsEntry), cudaMemcpyDeviceToHost));

    auto numEntries = readControl().numCompactedLineageEntries;
    result.lineageEntries.resize(numEntries);
    if (numEntries > 0) {
        CHECK_FOR_DEVICE_ERRORS(
            cudaMemcpy(result.lineageEntries.data(), _lineageStatisticsEntries, sizeof(LineageStatisticsEntry) * numEntries, cudaMemcpyDeviceToHost));
    }
    std::ranges::sort(result.lineageEntries, {}, &LineageStatisticsEntry::lineageId);
    return result;
}

bool SimulationStatistics::isLineageAccumulatorGCNeeded() const
{
    auto numUsedSlots = _accumulatorMaps[readControl().activeAccumulatorMapIndex].readNumUsedSlots();
    return numUsedSlots > static_cast<uint32_t>(AccumulatorMap::Capacity * AccumulatorGCLoadFactor);
}

SimulationStatistics::StatisticsControl SimulationStatistics::readControl() const
{
    StatisticsControl result;
    CHECK_FOR_DEVICE_ERRORS(cudaMemcpy(&result, _control, sizeof(StatisticsControl), cudaMemcpyDeviceToHost));
    return result;
}
