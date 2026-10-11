#pragma once

#include <cstdint>
#include <optional>
#include <string>

struct CommandLineArguments
{
    std::string inputFilename;
    std::string outputFilename;
    std::optional<uint64_t> timesteps;
    std::string userName;
    std::string password;
    std::string uploadName;
    std::optional<int> uploadInterval;
    bool debugMode = false;
    bool plainOutput = false;
};
