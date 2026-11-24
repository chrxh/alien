#pragma once

#include <chrono>
#include <string>

struct McpCommandLogEntry
{
    std::chrono::system_clock::time_point time;
    std::string toolName;
    std::string arguments;
    std::string result;
    bool isError = false;
};
