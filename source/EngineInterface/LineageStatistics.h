#pragma once

#include <cstdint>
#include <vector>

struct LineageStatisticsEntry
{
    uint32_t lineageId = 0;
    uint32_t colorBitset = 0;
    uint32_t numCreatures = 0;
    uint32_t numGenomes = 0;
    float sumCreatureCells = 0;
    float sumCreatureGenerations = 0;
    float sumGenomeNodes = 0;
    float sumMutationRates = 0;
    float sumCreatureEnergy = 0;

    uint64_t numCreatedCreatures = 0;  // Accumulated, never reset
    double totalMutations = 0;          // Accumulated, never reset
};

struct LineageStatistics
{
    std::vector<LineageStatisticsEntry> entries;
};
