#pragma once

#include <cstdint>
#include <vector>

struct ObjectStatisticsEntry
{
    uint32_t numSolidObjects = 0;
    uint32_t numFluidObjects = 0;
    uint32_t numFreeCellObjects = 0;
    uint32_t numCellObjects = 0;
    uint32_t numEnergyParticles = 0;
    double totalInternalEnergy = 0;  // Energy of all objects and energy particles, i.e. everything but the external energy pool
};

struct LineageStatisticsEntry
{
    uint32_t lineageId = 0;
    uint32_t colorBitset = 0;
    uint32_t numCreatures = 0;
    uint32_t numGenomes = 0;
    uint64_t sumCreatureCells = 0;
    uint64_t sumCreatureGenerations = 0;
    uint64_t sumGenomeNodes = 0;
    double sumMutationRates = 0;
    double sumCreatureEnergy = 0;
    uint64_t representativeCellId = 0;  // Cell of a creature with the highest generation; 0 = not set

    uint64_t numCreatedCreatures = 0;  // Accumulated, never reset
    double totalMutations = 0;         // Accumulated, never reset
    double totalAttackedEnergy = 0;    // Accumulated energy captured by attacker cells, never reset
    double totalMuscleActivity = 0;    // Accumulated lengths of the velocity changes caused by muscle cells, never reset
};

struct StatisticsEntry
{
    ObjectStatisticsEntry objectStatistics;
    std::vector<LineageStatisticsEntry> lineageEntries;  // Per-lineage statistics, sorted by lineageId
};
