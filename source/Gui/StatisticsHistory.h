#pragma once

#include <vector>

#include "EngineInterface/Definitions.h"

struct LiveStatistics
{
    static float constexpr MaxLiveHistory = 120.0f;  //in seconds

    float timepoint = 0.0f;  //in seconds
    float history = 10.0f;   //in seconds
    std::vector<float> timepointsHistory;
    std::vector<float> numCellsHistory;
    std::vector<float> numParticlesHistory;
    std::vector<float> numTokensHistory;
    std::vector<float> numCreatedCellsHistory;
    std::vector<float> numSuccessfulAttacksHistory;
    std::vector<float> numFailedAttacksHistory;
    std::vector<float> numMuscleActivitiesHistory;

    void truncate();
    void add(OverallStatistics const& statistics);
};

struct LongtermStatistics
{
    static float constexpr LongtermTimestepDelta = 1000.0f;

    std::vector<float> timestepHistory;
    std::vector<float> numCellsHistory;
    std::vector<float> numParticlesHistory;
    std::vector<float> numTokensHistory;
    std::vector<float> numCreatedCellsHistory;
    std::vector<float> numSuccessfulAttacksHistory;
    std::vector<float> numFailedAttacksHistory;
    std::vector<float> numMuscleActivitiesHistory;

    void add(OverallStatistics const& statistics);
};
