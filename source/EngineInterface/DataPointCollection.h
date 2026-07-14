#pragma once

#include <cstdint>
#include <unordered_map>

struct OverallDataPoint
{
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

    OverallDataPoint operator+(OverallDataPoint const& other) const;
    OverallDataPoint operator/(double divisor) const;
};

struct LineageDataPoint
{
    uint32_t colorBitset = 0;
    double numCreatures = 0;
    double numGenomes = 0;
    double sumCreatureCells = 0;
    double sumCreatureGenerations = 0;
    double sumGenomeNodes = 0;
    double sumMutationRates = 0;
    double sumCreatureEnergy = 0;

    double numCreatedCreatures = 0;
    double totalMutations = 0;
};

struct OverallDataPointCollection
{
    double time = 0;
    double systemClock = 0;

    OverallDataPoint overall;

    OverallDataPointCollection operator+(OverallDataPointCollection const& other) const;
    OverallDataPointCollection operator/(double divisor) const;
};

struct DataPointCollection
{
    double time = 0;
    double timestep = 0;
    double systemClock = 0;

    OverallDataPoint overall;
    std::unordered_map<uint32_t, LineageDataPoint> lineages;

    OverallDataPointCollection toOverallDataPointCollection() const;
};
