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

    void set(DeserializedSimulation&& value)
    {
        std::lock_guard lock(_mutex);
        _deserializedSimulation = std::move(value);
    }

    DeserializedSimulation get() const
    {
        std::lock_guard lock(_mutex);
        return _deserializedSimulation;
    }

private:
    mutable std::mutex _mutex;
    DeserializedSimulation _deserializedSimulation;
};
using SharedDeserializedSimulation = std::shared_ptr<_SharedDeserializedSimulation>;