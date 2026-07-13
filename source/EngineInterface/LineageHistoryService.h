#pragma once

#include <cstdint>

#include "LineageHistory.h"
#include "LineageStatistics.h"

class LineageHistoryService
{
public:
    void addSample(LineageHistory& history, LineageStatistics const& lineageStatistics, uint64_t timestep);
    void reset();

    // Merges two consecutive samples: union of the lineage ids, momentary values are averaged
    // and accumulated values are maximized where a lineage is present in both samples.
    static LineageSample mergeSamples(LineageSample const& earlierSample, LineageSample const& laterSample);

private:
    static auto constexpr MaxSamples = 1000;
    static auto constexpr DefaultTimestepDelta = 10.0;

    double _longtermTimestepDelta = DefaultTimestepDelta;
};
