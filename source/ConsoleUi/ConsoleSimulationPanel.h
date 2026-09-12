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
    uint32_t numEnergyParticles = 0;
    uint32_t numCreatures = 0;
    uint32_t numLineages = 0;

    std::vector<float> tpsHistory;

    void updateHistory();
};

class ConsoleSimulationPanel
{
public:
    static bool fitsIntoConsole();

    static std::vector<std::string> create(std::string const& title, ConsoleSimulationStatus const& status);
    static std::string createPlainLine(ConsoleSimulationStatus const& status);
};
