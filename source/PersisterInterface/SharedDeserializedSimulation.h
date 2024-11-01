#pragma once

#include <mutex>

#include "DeserializedSimulation.h"

class _SharedDeserializedSimulation
{
public:
    DataPointCollection getLastStatisticsData() const
    {
        std::lock_guard lock(_mutex);
        if (_deserializedSimulation.statistics.empty()) {
            return DataPointCollection();
        }
        return _deserializedSimulation.statistics.back();
    }

    void setDeserializedSimulation(DeserializedSimulation&& value)
    {
        std::lock_guard lock(_mutex);
        _deserializedSimulation = std::move(value);
    }

private:
    mutable std::mutex _mutex;
    DeserializedSimulation _deserializedSimulation;
};
using SharedDeserializedSimulation = std::shared_ptr<_SharedDeserializedSimulation>;