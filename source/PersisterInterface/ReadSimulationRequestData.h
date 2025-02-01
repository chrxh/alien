#pragma once

#include <filesystem>

struct ReadSimulationRequestData
{
    std::filesystem::path filename;
    bool initSimulation = false;    // Works only during startup
};