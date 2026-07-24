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

    for (auto const& entry : statisticsEntry.lineageEntries) {
        LineageDataPoint point;
        point.colorBitset = entry.colorBitset;
        point.representativeCellId = entry.representativeCellId;
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

        //aggregate the lineage into the overall bucket of its colorBitset; the truly global fields (solid/fluid/
        //energy-particle counts, active lineage count) are not per-color and are read live from the engine instead
        auto& colorPoint = result.overall[entry.colorBitset];
        colorPoint.numCreatures += point.numCreatures;
        colorPoint.numGenomes += point.numGenomes;
        colorPoint.sumCreatureCells += point.sumCreatureCells;
        colorPoint.sumCreatureGenerations += point.sumCreatureGenerations;
        colorPoint.sumGenomeNodes += point.sumGenomeNodes;
        colorPoint.sumMutationRates += point.sumMutationRates;
        colorPoint.sumCreatureEnergy += point.sumCreatureEnergy;
        colorPoint.numCreatedCreatures += point.numCreatedCreatures;
        colorPoint.totalMutations += point.totalMutations;
    }

    return result;
}
