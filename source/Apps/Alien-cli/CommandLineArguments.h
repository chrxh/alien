#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <Network/McpServer.h>

struct CommandLineArguments
{
    std::string inputFilename;
    std::string outputFilename;
    std::optional<uint64_t> timesteps;
    std::string userName;
    std::string password;
    std::string uploadName;
    std::optional<int> uploadInterval;
    bool mcpServer = false;
    int mcpPort = McpServer::DefaultPort;
    bool debugMode = false;
    bool plainOutput = false;
    bool noCountAsActive = false;
};
