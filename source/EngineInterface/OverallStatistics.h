#pragma once

#include <cstdint>

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
    uint32_t numActiveLineages = 0;
    uint32_t lineageMapOverflow = 0;
    uint32_t numCreatedCreatures = 0;  // Accumulated, never reset
    float totalMutations = 0;          // Accumulated, never reset
};
