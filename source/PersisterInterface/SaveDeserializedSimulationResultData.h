#pragma once

#include <chrono>
#include <filesystem>

struct SaveDeserializedSimulationResultData
{
    std::filesystem::path filename;
    std::string projectName;
    uint64_t timestep = 0;
    std::chrono::system_clock::time_point timestamp;
    StatisticsRawData statisticsRawData;
};
