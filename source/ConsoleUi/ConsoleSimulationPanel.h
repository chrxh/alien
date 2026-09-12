#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ConsoleSimulationStatus
{
    uint64_t timestep = 0;
    std::optional<uint64_t> totalTimesteps;  // Set for a run with a known end, which adds a progress bar
    std::chrono::milliseconds duration = std::chrono::milliseconds(0);
    float tps = 0.0f;
    bool paused = false;

    uint32_t numCells = 0;
    uint32_t numCreatures = 0;
    uint32_t numLineages = 0;
};

class ConsoleSimulationPanel
{
public:
    static bool fitsIntoConsole();

    static std::vector<std::string> create(ConsoleSimulationStatus const& status);
    static std::string createPlainLine(ConsoleSimulationStatus const& status);
};
