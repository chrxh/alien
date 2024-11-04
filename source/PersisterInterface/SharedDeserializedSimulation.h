#pragma once

#include <mutex>
#include <chrono>

#include "DeserializedSimulation.h"

class _SharedDeserializedSimulation
{
public:
    void setLastStatisticsData(RawStatisticsData const& value)
    {
        std::lock_guard lock(_mutex);
        _rawStatisticsData = value;
    }

    RawStatisticsData getRawStatisticsData() const
    {
        std::lock_guard lock(_mutex);
        return _rawStatisticsData;
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

    bool isEmpty() const
    {
        std::lock_guard lock(_mutex);
        return _deserializedSimulation.mainData.isEmpty();
    }

private:
    mutable std::mutex _mutex;
    DeserializedSimulation _deserializedSimulation;
    RawStatisticsData _rawStatisticsData;
    std::chrono::system_clock::time_point _timestamp;
};
using SharedDeserializedSimulation = std::shared_ptr<_SharedDeserializedSimulation>;