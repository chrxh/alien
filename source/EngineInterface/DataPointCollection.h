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
    double numEnergyParticles = 0;

    double accumCreatedCreatures = 0;  // Raw accumulated value; rates are derived GUI-side
    double accumMutations = 0;         // Raw accumulated value; rates are derived GUI-side

    OverallDataPoint operator+(OverallDataPoint const& other) const;
    OverallDataPoint operator/(double divisor) const;
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

    OverallDataPoint overall;
    std::unordered_map<uint32_t, LineageDataPoint> lineages;

    DataPointCollection operator+(DataPointCollection const& other) const;
    DataPointCollection operator/(double divisor) const;
};
