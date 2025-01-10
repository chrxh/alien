#pragma once

#include <mutex>
#include <chrono>

#include "DeserializedSimulation.h"

class _SharedDeserializedSimulation
{
public:
    void setLastStatisticsData(StatisticsRawData const& value)
    {
        std::lock_guard lock(_mutex);
        _statisticsRawData = value;
    }

    StatisticsRawData getStatisticsRawData() const
    {
        std::lock_guard lock(_mutex);
        return _statisticsRawData;
    }

    void setDeserializedSimulation(DeserializedSimulation&& value)
    {
        std::lock_guard lock(_mutex);
        _deserializedSimulation = std::move(value);
        _timestamp = std::chrono::system_clock::now();
    }

    DeserializedSimulation getDeserializedSimulation() const
    {
        std::lock_guard lock(_mutex);
        return _deserializedSimulation;
    }

    std::chrono::system_clock::time_point getTimestamp() const
    {
        std::lock_guard lock(_mutex);
        return _timestamp;
    }

    void reset()
    {
        setDeserializedSimulation(DeserializedSimulation());
        setLastStatisticsData(StatisticsRawData());
    }

    bool isEmpty() const
    {
        std::lock_guard lock(_mutex);
        return _deserializedSimulation.mainData.isEmpty();
    }

private:
    mutable std::mutex _mutex;
    DeserializedSimulation _deserializedSimulation;
    StatisticsRawData _statisticsRawData;
    std::chrono::system_clock::time_point _timestamp;
};
using SharedDeserializedSimulation = std::shared_ptr<_SharedDeserializedSimulation>;