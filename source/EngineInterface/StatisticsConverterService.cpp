#include "StatisticsConverterService.h"

#include <chrono>

#include <Base/Definitions.h>

DataPointCollection StatisticsConverterService::convert(
    StatisticsEntry const& statisticsEntry,
    uint64_t timestep,
    double time)
{
    DataPointCollection result;
    result.time = time;
    result.timestep = toDouble(timestep);

    auto now = std::chrono::system_clock::now();
    auto unixEpoch = std::chrono::time_point<std::chrono::system_clock>();
    result.systemClock = toDouble(std::chrono::duration_cast<std::chrono::seconds>(now - unixEpoch).count());

    auto const& o = statisticsEntry.overallEntry;
    result.overall.numCreatures = toDouble(o.numCreatures);
    result.overall.averageCreatureCells = o.numCreatures > 0 ? toDouble(o.sumCreatureCells) / o.numCreatures : 0.0;
    result.overall.averageGeneration = o.numCreatures > 0 ? toDouble(o.sumCreatureGenerations) / o.numCreatures : 0.0;
    result.overall.averageGenomeNodes = o.numGenomes > 0 ? toDouble(o.sumGenomeNodes) / o.numGenomes : 0.0;
    result.overall.averageMutationRate = o.numGenomes > 0 ? toDouble(o.sumMutationRates) / o.numGenomes : 0.0;
    result.overall.creatureEnergy = toDouble(o.sumCreatureEnergy);
    result.overall.numLineages = toDouble(o.numActiveLineages);
    result.overall.numSolidObjects = toDouble(o.numSolidObjects);
    result.overall.numFluidObjects = toDouble(o.numFluidObjects);
    result.overall.numCellObjects = toDouble(o.numCellObjects);
    result.overall.accumCreatedCreatures = toDouble(o.numCreatedCreatures);
    result.overall.accumMutations = toDouble(o.totalMutations);

    for (auto const& entry : statisticsEntry.entries) {
        LineageDataPoint point;
        point.colorBitset = entry.colorBitset;
        point.numCreatures = toDouble(entry.numCreatures);
        point.numGenomes = toDouble(entry.numGenomes);
        point.sumCreatureCells = toDouble(entry.sumCreatureCells);
        point.sumCreatureGenerations = toDouble(entry.sumCreatureGenerations);
        point.sumGenomeNodes = toDouble(entry.sumGenomeNodes);
        point.sumMutationRates = toDouble(entry.sumMutationRates);
        point.sumCreatureEnergy = toDouble(entry.sumCreatureEnergy);
        point.numCreatedCreatures = toDouble(entry.numCreatedCreatures);
        point.totalMutations = entry.totalMutations;
        result.lineages[entry.lineageId] = point;
    }

    return result;
}
