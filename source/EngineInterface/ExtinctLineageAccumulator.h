#pragma once

#include <cstdint>
#include <unordered_map>

#include "DataPointCollection.h"

// The accumulated counters of the overall statistics (numCreatedCreatures, totalMutations) are summed up from the
// currently living lineages only, so they would drop as soon as a lineage becomes extinct although they are meant to
// increase monotonically. This class remembers the last known counters of extinct lineages and adds them back.
// Note: a lineage id that reappears after its extinction (only possible for imported objects) is counted twice.
class ExtinctLineageAccumulator
{
public:
    // Retires the lineages that are no longer present and adds the counters of all extinct lineages to the overall data
    void addExtinctLineageValues(DataPointCollection& dataPoints);

    // Discards the extinct lineages, e.g. after a different simulation has been loaded
    void reset();

private:
    struct AccumulatedValues
    {
        double numCreatedCreatures = 0;
        double totalMutations = 0;
    };

    struct LastLineageValues
    {
        uint32_t colorBitset = 0;
        AccumulatedValues values;
    };

    std::unordered_map<uint32_t, LastLineageValues> _lastLineageValues;  // Key: lineage id
    std::unordered_map<uint32_t, AccumulatedValues> _extinctValues;      // Key: color bitset
};
