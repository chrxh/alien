#pragma once

#include <vector>

#include "EngineInterface/Definitions.h"

struct LiveStatistics
{
    static float constexpr MaxLiveHistory = 120.0f;  //in seconds

    float timepoint = 0.0f;  //in seconds
    float history = 10.0f;   //in seconds

    std::vector<float> timepointsHistory;
    std::array<std::vector<float>, 15>
        datas;  //cells, cells by colors (7x), particles, tokens, created cells, successful attacks, failed attacks, muscle activities

    void truncate();
    void add(MonitorData const& statistics);
};

struct LongtermStatistics
{
    static float constexpr LongtermTimestepDelta = 1000.0f;

    std::vector<float> timestepHistory;
    std::array<std::vector<float>, 15>
        datas;  //cells, cells by colors (7x), particles, tokens, created cells, successful attacks, failed attacks, muscle activities

    void add(MonitorData const& statistics);
};
