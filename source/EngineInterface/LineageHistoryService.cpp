#include "LineageHistoryService.h"

#include <algorithm>
#include <chrono>

#include <Base/Definitions.h>

void LineageHistoryService::addSample(LineageHistory& history, LineageStatistics const& lineageStatistics, uint64_t timestep)
{
    std::lock_guard lock(history.getMutex());
    auto& historyData = history.getDataRef();

    if (!historyData.empty() && historyData.back().time > toDouble(timestep) + NEAR_ZERO) {
        historyData.clear();
        _longtermTimestepDelta = DefaultTimestepDelta;
    }

    if (!historyData.empty() && toDouble(timestep) - historyData.back().time < _longtermTimestepDelta) {
        return;
    }

    LineageSample sample;
    sample.time = toDouble(timestep);
    auto now = std::chrono::system_clock::now();
    auto unixEpoch = std::chrono::time_point<std::chrono::system_clock>();
    sample.systemClock = toDouble(std::chrono::duration_cast<std::chrono::seconds>(now - unixEpoch).count());
    sample.entries = lineageStatistics.entries;
    std::sort(sample.entries.begin(), sample.entries.end(), [](auto const& lhs, auto const& rhs) { return lhs.lineageId < rhs.lineageId; });
    historyData.emplace_back(std::move(sample));

    if (historyData.size() > MaxSamples) {
        LineageHistoryData newData;
        newData.reserve(historyData.size() / 2 + 1);
        for (size_t i = 0; i < (historyData.size() - 1) / 2; ++i) {
            newData.emplace_back(mergeSamples(historyData.at(i * 2), historyData.at(i * 2 + 1)));
        }
        newData.emplace_back(historyData.back());
        historyData.swap(newData);

        _longtermTimestepDelta *= 2.0;
    }
}

void LineageHistoryService::reset()
{
    _longtermTimestepDelta = DefaultTimestepDelta;
}

LineageSample LineageHistoryService::mergeSamples(LineageSample const& earlierSample, LineageSample const& laterSample)
{
    LineageSample result;
    result.time = earlierSample.time;
    result.systemClock = (earlierSample.systemClock + laterSample.systemClock) / 2;
    result.entries.reserve(earlierSample.entries.size() + laterSample.entries.size());

    auto mergeEntries = [](LineageStatisticsEntry const& lhs, LineageStatisticsEntry const& rhs) {
        LineageStatisticsEntry result;
        result.lineageId = lhs.lineageId;
        result.colorBitset = lhs.colorBitset | rhs.colorBitset;
        result.numCreatures = (lhs.numCreatures + rhs.numCreatures) / 2;
        result.numGenomes = (lhs.numGenomes + rhs.numGenomes) / 2;
        result.sumCreatureCells = (lhs.sumCreatureCells + rhs.sumCreatureCells) / 2;
        result.sumCreatureGenerations = (lhs.sumCreatureGenerations + rhs.sumCreatureGenerations) / 2;
        result.sumGenomeNodes = (lhs.sumGenomeNodes + rhs.sumGenomeNodes) / 2;
        result.sumMutationRates = (lhs.sumMutationRates + rhs.sumMutationRates) / 2;
        result.sumCreatureEnergy = (lhs.sumCreatureEnergy + rhs.sumCreatureEnergy) / 2;
        result.numCreatedCreatures = std::max(lhs.numCreatedCreatures, rhs.numCreatedCreatures);
        result.totalMutations = std::max(lhs.totalMutations, rhs.totalMutations);
        return result;
    };

    auto earlierIter = earlierSample.entries.begin();
    auto laterIter = laterSample.entries.begin();
    while (earlierIter != earlierSample.entries.end() || laterIter != laterSample.entries.end()) {
        if (laterIter == laterSample.entries.end() || (earlierIter != earlierSample.entries.end() && earlierIter->lineageId < laterIter->lineageId)) {
            result.entries.emplace_back(*earlierIter);
            ++earlierIter;
        } else if (earlierIter == earlierSample.entries.end() || laterIter->lineageId < earlierIter->lineageId) {
            result.entries.emplace_back(*laterIter);
            ++laterIter;
        } else {
            result.entries.emplace_back(mergeEntries(*earlierIter, *laterIter));
            ++earlierIter;
            ++laterIter;
        }
    }
    return result;
}
