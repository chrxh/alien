#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "DataPointCollection.h"

template <typename DataPoint>
struct TimedSample
{
    double timestep = 0;
    double systemClock = 0;
    DataPoint data;

    TimedSample operator+(TimedSample const& other) const
    {
        TimedSample result;
        result.timestep = timestep + other.timestep;
        result.systemClock = systemClock + other.systemClock;
        result.data = data + other.data;
        return result;
    }

    TimedSample operator/(double divisor) const
    {
        TimedSample result;
        result.timestep = timestep / divisor;
        result.systemClock = systemClock / divisor;
        result.data = data / divisor;
        return result;
    }
};

template <>
struct TimedSample<std::unordered_map<uint32_t, ColorOverallDataPoint>>
{
    double timestep = 0;
    double systemClock = 0;
    std::unordered_map<uint32_t, ColorOverallDataPoint> data;

    TimedSample operator+(TimedSample const& other) const
    {
        TimedSample result;
        result.timestep = timestep + other.timestep;
        result.systemClock = systemClock + other.systemClock;
        result.data = data;
        for (auto const& [colorBitset, dataPoint] : other.data) {
            auto it = result.data.find(colorBitset);
            if (it != result.data.end()) {
                it->second = it->second + dataPoint;
            } else {
                result.data.emplace(colorBitset, dataPoint);
            }
        }
        return result;
    }

    TimedSample operator/(double divisor) const
    {
        TimedSample result;
        result.timestep = timestep / divisor;
        result.systemClock = systemClock / divisor;
        result.data = data;
        for (auto& [colorBitset, dataPoint] : result.data) {
            dataPoint = dataPoint / divisor;
        }
        return result;
    }
};

using ColorSamples = TimedSample<std::unordered_map<uint32_t, ColorOverallDataPoint>>;
using LineageSample = TimedSample<LineageDataPoint>;

// The overall data and each lineage have separate timelines so that sampling density and compression
// can be controlled independently: young lineages keep a high sampling rate even in old simulations
struct StatisticsHistoryData
{
    std::vector<ColorSamples> colors;
    std::unordered_map<uint32_t, std::vector<LineageSample>> lineages;
};

class StatisticsHistory
{
public:
    StatisticsHistoryData getCopiedData() const;

    std::mutex& getMutex() const;
    StatisticsHistoryData& getDataRef();
    StatisticsHistoryData const& getDataRef() const;

private:
    mutable std::mutex _mutex;
    StatisticsHistoryData _data;
};
