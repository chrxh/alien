#pragma once

#include <cstdint>
#include <unordered_map>

// Summable statistics of all creatures sharing one colorBitset; the fields mirror the aggregatable part of
// LineageDataPoint (everything except colorBitset and representativeCellId) so that the same metric derivation
// serves single lineages and color-filtered overall statistics
struct ColorOverallDataPoint
{
    double numCreatures = 0;
    double numGenomes = 0;
    double sumCreatureCells = 0;
    double sumCreatureGenerations = 0;
    double sumGenomeNodes = 0;
    double sumMutationRates = 0;
    double sumCreatureEnergy = 0;
    double numCreatedCreatures = 0;
    double totalMutations = 0;

    ColorOverallDataPoint operator+(ColorOverallDataPoint const& other) const;
    ColorOverallDataPoint operator/(double divisor) const;
};

struct LineageDataPoint
{
    uint32_t colorBitset = 0;
    uint64_t representativeCellId = 0;  // Cell of a creature with the highest generation; only meaningful in recent samples
    double numCreatures = 0;
    double numGenomes = 0;
    double sumCreatureCells = 0;
    double sumCreatureGenerations = 0;
    double sumGenomeNodes = 0;
    double sumMutationRates = 0;
    double sumCreatureEnergy = 0;

    double numCreatedCreatures = 0;
    double totalMutations = 0;

    LineageDataPoint operator+(LineageDataPoint const& other) const;
    LineageDataPoint operator/(double divisor) const;
};

struct DataPointCollection
{
    double time = 0;
    double timestep = 0;
    double systemClock = 0;

    // Overall statistics broken down by colorBitset; overall statistics for a color filter are assembled by summing
    // the buckets whose colorBitset intersects the filter (each creature falls into exactly one bucket)
    std::unordered_map<uint32_t, ColorOverallDataPoint> overall;
    std::unordered_map<uint32_t, LineageDataPoint> lineages;

    DataPointCollection operator+(DataPointCollection const& other) const;
    DataPointCollection operator/(double divisor) const;
};
