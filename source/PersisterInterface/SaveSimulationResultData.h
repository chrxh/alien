#pragma once

#include <chrono>
#include <string>

struct SaveSimulationResultData
{
    std::filesystem::path filename;
    std::string name;
    uint64_t timestep = 0;
    std::chrono::system_clock::time_point timestamp;
};
