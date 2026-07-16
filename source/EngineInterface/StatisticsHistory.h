#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "DataPointCollection.h"

template <typename DataPoint>
struct TimedSample
{
    double time = 0;         // Time since simulation start
    double timestep = 0;
    double systemClock = 0;
    DataPoint data;

    TimedSample operator+(TimedSample const& other) const
    {
        TimedSample result;
        result.time = time + other.time;
        result.timestep = timestep + other.timestep;
        result.systemClock = systemClock + other.systemClock;
        result.data = data + other.data;
        return result;
    }

    TimedSample operator/(double divisor) const
    {
        TimedSample result;
        result.time = time / divisor;
        result.timestep = timestep / divisor;
        result.systemClock = systemClock / divisor;
        result.data = data / divisor;
        return result;
    }
};

using OverallSample = TimedSample<OverallDataPoint>;
using LineageSample = TimedSample<LineageDataPoint>;

//the overall data and each lineage have separate timelines so that sampling density and compression
//can be controlled independently: young lineages keep a high sampling rate even in old simulations
struct StatisticsHistoryData
{
    std::vector<OverallSample> overall;
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
