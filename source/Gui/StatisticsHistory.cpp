#include "StatisticsHistory.h"

#include "EngineInterface/OverallStatistics.h"

#include <imgui.h>

void LiveStatistics::truncate()
{
    if (!timepointsHistory.empty() && timepointsHistory.back() - timepointsHistory.front() > (MaxLiveHistory + 1.0f)) {
        timepointsHistory.erase(timepointsHistory.begin());
        numCellsHistory.erase(numCellsHistory.begin());
        numParticlesHistory.erase(numParticlesHistory.begin());
        numTokensHistory.erase(numTokensHistory.begin());
        numCreatedCellsHistory.erase(numCreatedCellsHistory.begin());
        numSuccessfulAttacksHistory.erase(numSuccessfulAttacksHistory.begin());
        numFailedAttacksHistory.erase(numFailedAttacksHistory.begin());
        numMuscleActivitiesHistory.erase(numMuscleActivitiesHistory.begin());
    }
}

void LiveStatistics::add(OverallStatistics const& newStatistics)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;
    timepointsHistory.emplace_back(timepoint);
    numCellsHistory.emplace_back(toFloat(newStatistics.numCells));
    numParticlesHistory.emplace_back(toFloat(newStatistics.numParticles));
    numTokensHistory.emplace_back(toFloat(newStatistics.numTokens));
    numCreatedCellsHistory.emplace_back(toFloat(newStatistics.numCreatedCells));
    numSuccessfulAttacksHistory.emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
    numFailedAttacksHistory.emplace_back(toFloat(newStatistics.numFailedAttacks));
    numMuscleActivitiesHistory.emplace_back(toFloat(newStatistics.numMuscleActivities));
}

void LongtermStatistics::add(OverallStatistics const& newStatistics)
{
    if (timestepHistory.empty() || newStatistics.timeStep - timestepHistory.back() > LongtermTimestepDelta) {
        timestepHistory.emplace_back(toFloat(newStatistics.timeStep));
        numCellsHistory.emplace_back(toFloat(newStatistics.numCells));
        numParticlesHistory.emplace_back(toFloat(newStatistics.numParticles));
        numTokensHistory.emplace_back(toFloat(newStatistics.numTokens));
        numCreatedCellsHistory.emplace_back(toFloat(newStatistics.numCreatedCells));
        numSuccessfulAttacksHistory.emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
        numFailedAttacksHistory.emplace_back(toFloat(newStatistics.numFailedAttacks));
        numMuscleActivitiesHistory.emplace_back(toFloat(newStatistics.numMuscleActivities));
    }
}
