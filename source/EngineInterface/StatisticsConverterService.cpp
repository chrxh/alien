#include "StatisticsConverterService.h"

#include <chrono>

#include <Base/Definitions.h>

DataPointCollection StatisticsConverterService::convert(
    OverallStatisticsEntry const& overallStatistics,
    uint64_t timestep,
    double time)
{
    DataPointCollection result;
    result.time = time;

    auto now = std::chrono::system_clock::now();
    auto unixEpoch = std::chrono::time_point<std::chrono::system_clock>();
    result.systemClock = toDouble(std::chrono::duration_cast<std::chrono::seconds>(now - unixEpoch).count());

    auto const& overall = overallStatistics;
    result.numCreatures = toDouble(overall.numCreatures);
    result.averageCreatureCells = overall.numCreatures > 0 ? toDouble(overall.sumCreatureCells) / overall.numCreatures : 0.0;
    result.averageGeneration = overall.numCreatures > 0 ? toDouble(overall.sumCreatureGenerations) / overall.numCreatures : 0.0;
    result.averageGenomeNodes = overall.numGenomes > 0 ? toDouble(overall.sumGenomeNodes) / overall.numGenomes : 0.0;
    result.averageMutationRate = overall.numGenomes > 0 ? toDouble(overall.sumMutationRates) / overall.numGenomes : 0.0;
    result.creatureEnergy = toDouble(overall.sumCreatureEnergy);
    result.numLineages = toDouble(overall.numActiveLineages);
    result.numSolidObjects = toDouble(overall.numSolidObjects);
    result.numFluidObjects = toDouble(overall.numFluidObjects);
    result.numCellObjects = toDouble(overall.numCellObjects);
    result.accumCreatedCreatures = toDouble(overall.numCreatedCreatures);
    result.accumMutations = toDouble(overall.totalMutations);

    return result;
}
