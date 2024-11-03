#pragma once

struct SaveDeserializedSimulationResultData
{
    std::filesystem::path filename;
    std::string projectName;
    uint64_t timestep = 0;
    std::chrono::system_clock::time_point timestamp;
};
