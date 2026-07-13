#pragma once

#include <EngineInterface/Colors.h>
#include <EngineInterface/EngineConstants.h>

struct EvolutionStatistics
{
    int numCreatures = 0;
    float sumCreatureCells = 0;
    float sumCreatureGenerations = 0;
    float sumAccumulatedMutations = 0;
    int numGenomes = 0;
    float sumGenomeNodes = 0;
    float sumMutationRates = 0;  // Sum over genomes of the per-genome mean of all mutation probabilities
    float sumCreatureEnergy = 0;
    int numSolidObjects = 0;
    int numFluidObjects = 0;
    int numCellObjects = 0;
    int numActiveLineages = 0;
    int lineageMapOverflow = 0;
};

struct TimestepStatistics
{
    ColorVector<int> numObjects = {};
    ColorVector<int> numSelfReplicators = {};
    ColorVector<int> numColonies = {};
    ColorVector<int> numViruses = {};
    ColorVector<int> numFreeCells = {};
    ColorVector<int> numEnergyParticles = {};
    ColorVector<float> totalEnergy = {};
    EvolutionStatistics evolution;
};

struct AccumulatedStatistics
{
    ColorVector<uint64_t> numCreatedCells = {};
    ColorVector<uint64_t> numCreatedReplicators = {};
    ColorVector<uint64_t> numAttacks = {};
    ColorVector<uint64_t> numMuscleActivities = {};
    ColorVector<uint64_t> numDefenderActivities = {};
    ColorVector<uint64_t> numDepotActivities = {};
    ColorVector<uint64_t> numInjectionActivities = {};
    ColorVector<uint64_t> numCompletedInjections = {};
    ColorVector<uint64_t> numGeneratorPulses = {};
    ColorVector<uint64_t> numNeuronActivities = {};
    ColorVector<uint64_t> numSensorActivities = {};
    ColorVector<uint64_t> numSensorMatches = {};
    ColorVector<uint64_t> numReconnectorCreated = {};
    ColorVector<uint64_t> numReconnectorRemoved = {};
    ColorVector<uint64_t> numDetonations = {};
    uint64_t numCreatedCreatures = 0;
    double totalMutations = 0;
};

struct TimelineStatistics
{
    TimestepStatistics timestep;
    AccumulatedStatistics accumulated;
};

struct HistogramData
{
    int maxAge = 0;
    int numCellsByColorBySlot[MAX_COLORS][MAX_HISTOGRAM_SLOTS];
};

struct StatisticsRawData
{
    TimelineStatistics timeline;
    HistogramData histogram;
};

inline double sumColorVector(ColorVector<double> const& v)
{
    auto result = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        result += v[i];
    }
    return result;
};
