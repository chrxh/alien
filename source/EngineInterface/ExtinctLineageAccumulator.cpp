#include "ExtinctLineageAccumulator.h"

void ExtinctLineageAccumulator::addExtinctLineageValues(DataPointCollection& dataPoints)
{
    // Archive lineages that disappeared
    for (auto lastIt = _lastLineageValues.begin(); lastIt != _lastLineageValues.end();) {
        if (dataPoints.lineages.contains(lastIt->first)) {
            ++lastIt;
            continue;
        }
        auto& extinctValues = _extinctValues[lastIt->second.colorBitset];
        extinctValues.numCreatedCreatures += lastIt->second.values.numCreatedCreatures;
        extinctValues.totalMutations += lastIt->second.values.totalMutations;
        lastIt = _lastLineageValues.erase(lastIt);
    }

    // Remember the current values of the living lineages
    for (auto const& [lineageId, dataPoint] : dataPoints.lineages) {
        auto& lastLineageValues = _lastLineageValues[lineageId];
        lastLineageValues.colorBitset = dataPoint.colorBitset;
        lastLineageValues.values.numCreatedCreatures = dataPoint.numCreatedCreatures;
        lastLineageValues.values.totalMutations = dataPoint.totalMutations;
    }

    // Add the archived values back to dataPoints
    for (auto const& [colorBitset, extinctValues] : _extinctValues) {
        auto& colorPoint = dataPoints.overall[colorBitset];
        colorPoint.numCreatedCreatures += extinctValues.numCreatedCreatures;
        colorPoint.totalMutations += extinctValues.totalMutations;
    }
}

void ExtinctLineageAccumulator::reset()
{
    _lastLineageValues.clear();
    _extinctValues.clear();
}
