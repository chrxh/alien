#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct CommandLineArguments
{
    std::string inputFilename;
    std::string outputFilename;
    std::optional<uint64_t> timesteps;
    std::vector<int> gpus;
    std::string userName;
    std::string password;
    std::string uploadName;
    std::optional<int> uploadInterval;
    bool debugMode = false;
    bool plainOutput = false;
};
