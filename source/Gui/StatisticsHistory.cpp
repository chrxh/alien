#include "StatisticsHistory.h"

#include "EngineInterface/OverallStatistics.h"

#include <imgui.h>

void LiveStatistics::truncate()
{
    if (!timepointsHistory.empty() && timepointsHistory.back() - timepointsHistory.front() > (MaxLiveHistory + 1.0f)) {
        timepointsHistory.erase(timepointsHistory.begin());
        for(auto& data : datas) {
            data.erase(data.begin());
        }
    }
}

void LiveStatistics::add(OverallStatistics const& newStatistics)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;
    timepointsHistory.emplace_back(timepoint);
    datas[0].emplace_back(toFloat(newStatistics.numCells));
    datas[1].emplace_back(toFloat(newStatistics.numParticles));
    datas[2].emplace_back(toFloat(newStatistics.numTokens));
    datas[3].emplace_back(toFloat(newStatistics.numCreatedCells));
    datas[4].emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
    datas[5].emplace_back(toFloat(newStatistics.numFailedAttacks));
    datas[6].emplace_back(toFloat(newStatistics.numMuscleActivities));
}

void LongtermStatistics::add(OverallStatistics const& newStatistics)
{
    if (timestepHistory.empty() || newStatistics.timeStep - timestepHistory.back() > LongtermTimestepDelta) {
        timestepHistory.emplace_back(toFloat(newStatistics.timeStep));
        datas[0].emplace_back(toFloat(newStatistics.numCells));
        datas[1].emplace_back(toFloat(newStatistics.numParticles));
        datas[2].emplace_back(toFloat(newStatistics.numTokens));
        datas[3].emplace_back(toFloat(newStatistics.numCreatedCells));
        datas[4].emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
        datas[5].emplace_back(toFloat(newStatistics.numFailedAttacks));
        datas[6].emplace_back(toFloat(newStatistics.numMuscleActivities));
    }
}
