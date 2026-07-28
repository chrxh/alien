#pragma once

#include <chrono>
#include <mutex>

#include <EngineInterface/Descs.h>

class _SharedDeserializedSimulation
{
public:
    void setDeserializedSimulation(SimulationDesc&& value)
    {
        std::lock_guard lock(_mutex);
        _deserializedSimulation = std::move(value);
        _timestamp = std::chrono::system_clock::now();
    }

    SimulationDesc getDeserializedSimulation() const
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
        setDeserializedSimulation(SimulationDesc());
    }

    bool isEmpty() const
    {
        std::lock_guard lock(_mutex);
        return _deserializedSimulation._mainData.isEmpty();
    }

private:
    mutable std::mutex _mutex;
    SimulationDesc _deserializedSimulation;
    std::chrono::system_clock::time_point _timestamp;
};
using SharedDeserializedSimulation = std::shared_ptr<_SharedDeserializedSimulation>;
