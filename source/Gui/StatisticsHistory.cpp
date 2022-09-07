#include "StatisticsHistory.h"

#include "EngineInterface/MonitorData.h"

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

void LiveStatistics::add(MonitorData const& newStatistics)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;
    timepointsHistory.emplace_back(timepoint);
    int numCells = 0;
    for (int i = 0; i < 7; ++i) {
        numCells += newStatistics.numCellsByColor[i];
    }
    datas[0].emplace_back(toFloat(numCells));
    for (int i = 0; i < 7; ++i) {
        datas[1 + i].emplace_back(toFloat(newStatistics.numCellsByColor[i]));
    }
    datas[8].emplace_back(toFloat(newStatistics.numConnections));
    datas[9].emplace_back(toFloat(newStatistics.numParticles));
    datas[10].emplace_back(toFloat(newStatistics.numTokens));
    datas[11].emplace_back(toFloat(newStatistics.numCreatedCells));
    datas[12].emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
    datas[13].emplace_back(toFloat(newStatistics.numFailedAttacks));
    datas[14].emplace_back(toFloat(newStatistics.numMuscleActivities));
}

void LongtermStatistics::add(MonitorData const& newStatistics)
{
    if (timestepHistory.empty() || newStatistics.timeStep - timestepHistory.back() > LongtermTimestepDelta) {
        timestepHistory.emplace_back(toFloat(newStatistics.timeStep));
        int numCells = 0;
        for (int i = 0; i < 7; ++i) {
            numCells += newStatistics.numCellsByColor[i];
        }
        datas[0].emplace_back(toFloat(numCells));
        for (int i = 0; i < 7; ++i) {
            datas[1 + i].emplace_back(toFloat(newStatistics.numCellsByColor[i]));
        }
        datas[8].emplace_back(toFloat(newStatistics.numConnections));
        datas[9].emplace_back(toFloat(newStatistics.numParticles));
        datas[10].emplace_back(toFloat(newStatistics.numTokens));
        datas[11].emplace_back(accumulatedCreatedCells);
        datas[12].emplace_back(accumulatedSuccessfulAttacks);
        datas[13].emplace_back(accumulatedFailedAttack);
        datas[14].emplace_back(accumulatedMuscleActivities);
        accumulatedCreatedCells = 0;
        accumulatedSuccessfulAttacks = 0;
        accumulatedFailedAttack = 0;
        accumulatedMuscleActivities = 0;
    } else {
        accumulatedCreatedCells += newStatistics.numCreatedCells;
        accumulatedSuccessfulAttacks += newStatistics.numSuccessfulAttacks;
        accumulatedFailedAttack += newStatistics.numFailedAttacks;
        accumulatedMuscleActivities += newStatistics.numMuscleActivities;
    }
}
