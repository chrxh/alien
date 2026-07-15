#pragma once

#include <cstdint>
#include <vector>

struct OverallStatisticsEntry
{
    uint32_t numCreatures = 0;
    uint32_t numGenomes = 0;
    float sumCreatureCells = 0;
    float sumCreatureGenerations = 0;
    float sumGenomeNodes = 0;
    float sumMutationRates = 0;
    float sumCreatureEnergy = 0;
    float sumAccumulatedMutations = 0;
    uint32_t numSolidObjects = 0;
    uint32_t numFluidObjects = 0;
    uint32_t numCellObjects = 0;
    uint32_t numEnergyParticles = 0;
    uint32_t numActiveLineages = 0;

    unsigned long long numCreatedCreatures = 0;  // Accumulated, never reset
    float totalMutations = 0;                    // Accumulated, never reset
};

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
    double totalMutations = 0;         // Accumulated, never reset
};

struct StatisticsEntry
{
    OverallStatisticsEntry overallEntry;
    std::vector<LineageStatisticsEntry> entries;  // Per-lineage statistics, sorted by lineageId
};
