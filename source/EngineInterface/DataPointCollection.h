#pragma once

#include <vector>

#include "StatisticsEntry.h"

struct DataPointCollection
{
    double time = 0;
    double systemClock = 0;

    // Evolution dashboard values
    double numCreatures = 0;
    double averageCreatureCells = 0;
    double averageGenomeNodes = 0;
    double creatureEnergy = 0;
    double averageMutationRate = 0;
    double averageGeneration = 0;
    double numLineages = 0;
    double numSolidObjects = 0;
    double numFluidObjects = 0;
    double numCellObjects = 0;
    double accumCreatedCreatures = 0;  // Raw accumulated value; rates are derived GUI-side
    double accumMutations = 0;         // Raw accumulated value; rates are derived GUI-side

    std::vector<LineageStatisticsEntry> lineageEntries;  // Per-lineage statistics, sorted by lineageId

    DataPointCollection operator+(DataPointCollection const& other) const;
    DataPointCollection operator/(double divisor) const;
};
