#include "SimulationRunner.h"

#include <algorithm>
#include <chrono>

#include <Base/ExitScopeGuard.h>

#include <ConsoleUi/ConsoleInput.h>
#include <ConsoleUi/ConsoleLiveOutput.h>
#include <ConsoleUi/ConsoleSimulationPanel.h>

#include <Engine/Interface/SimulationFacade.h>

#include <McpTools/Interface/McpToolsFacade.h>

#include "PeriodicUploader.h"

namespace
{
    auto constexpr StatusUpdateInterval = std::chrono::milliseconds(100);
    auto constexpr MaxChunkGrowth = 4;

    uint64_t calcNextChunkSize(uint64_t chunkSize, std::chrono::steady_clock::duration const& chunkDuration)
    {
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(chunkDuration).count();
        if (microseconds <= 0) {
            return chunkSize * MaxChunkGrowth;
        }
        auto targetMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(StatusUpdateInterval).count();
        auto scaled = static_cast<uint64_t>(toDouble(chunkSize) * toDouble(targetMicroseconds) / toDouble(microseconds));
        return std::clamp(scaled, uint64_t(1), chunkSize * MaxChunkGrowth);
    }

    bool isStopRequested()
    {
        auto pressedCharacter = ConsoleInput::readPressedCharacter();
        return ConsoleInput::isQuitRequested() || pressedCharacter == 'q' || pressedCharacter == 'Q';
    }

    void updateLiveOutput(ConsoleLiveOutput& liveOutput, ConsoleSimulationStatus& status, SimulationFacade const& simulationFacade)
    {
        auto statistics = simulationFacade->getStatisticsEntry();
        status.numCells = statistics.objectStatistics.numCellObjects;
        status.numCreatures = 0;
        for (auto const& lineage : statistics.lineageEntries) {
            status.numCreatures += lineage.numCreatures;
        }
        status.numLineages = toUInt32(statistics.lineageEntries.size());

        auto lines = ConsoleSimulationPanel::fitsIntoConsole() ? ConsoleSimulationPanel::create(status) : std::vector<std::string>();
        liveOutput.update(lines, ConsoleSimulationPanel::createPlainLine(status));
    }
}

uint64_t SimulationRunner::calcTimestepsWithLiveOutput(
    SimulationFacade const& simulationFacade,
    std::optional<uint64_t> timesteps,
    ConsoleLiveOutput& liveOutput,
    PeriodicUploader* periodicUploader)
{
    ConsoleSimulationStatus status;
    status.startTimestep = simulationFacade->getCurrentTimestep();
    status.timestep = status.startTimestep;
    if (timesteps.has_value()) {
        status.endTimestep = status.startTimestep + *timesteps;
    }
    auto startRealTime = simulationFacade->getRealTime();

    auto startTimepoint = std::chrono::steady_clock::now();
    auto lastUpdateTimepoint = startTimepoint;
    auto chunkSize = uint64_t(1);
    auto timestepsSinceUpdate = uint64_t(0);

    ConsoleInput::begin();
    while (!status.endTimestep.has_value() || status.timestep < *status.endTimestep) {
        auto chunk = status.endTimestep.has_value() ? std::min(chunkSize, *status.endTimestep - status.timestep) : chunkSize;

        auto chunkStartTimepoint = std::chrono::steady_clock::now();
        simulationFacade->calcTimesteps(chunk);
        auto now = std::chrono::steady_clock::now();

        status.timestep += chunk;
        timestepsSinceUpdate += chunk;
        chunkSize = calcNextChunkSize(chunkSize, now - chunkStartTimepoint);

        auto stopRequested = isStopRequested();
        auto finished = stopRequested || (status.endTimestep.has_value() && status.timestep >= *status.endTimestep);
        if (now - lastUpdateTimepoint < StatusUpdateInterval && !finished) {
            continue;
        }
        auto intervalMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(now - lastUpdateTimepoint).count();
        lastUpdateTimepoint = now;

        status.realTime = startRealTime + std::chrono::duration_cast<std::chrono::milliseconds>(now - startTimepoint);
        simulationFacade->setRealTime(status.realTime);
        status.tps = intervalMicroseconds > 0 ? toFloat(timestepsSinceUpdate) * 1.0e6f / toFloat(intervalMicroseconds) : 0.0f;
        timestepsSinceUpdate = 0;

        updateLiveOutput(liveOutput, status, simulationFacade);

        if (periodicUploader) {
            periodicUploader->process();
        }
        if (stopRequested) {
            break;
        }
    }
    if (periodicUploader) {
        periodicUploader->waitForPendingUpload();
    }
    ConsoleInput::end();
    liveOutput.close();
    return status.timestep - status.startTimestep;
}

void SimulationRunner::runWithMcpServer(SimulationFacade const& simulationFacade, ConsoleLiveOutput& liveOutput, PeriodicUploader* periodicUploader)
{
    ConsoleSimulationStatus status;
    std::optional<std::chrono::steady_clock::time_point> lastUpdateTimepoint;

    ConsoleInput::begin();
    ExitScopeGuard endConsoleInput(&ConsoleInput::end);
    while (!isStopRequested()) {
        simulationFacade->checkAndThrowException();
        _McpToolsFacade::get()->processToolCalls();

        auto now = std::chrono::steady_clock::now();
        if (!lastUpdateTimepoint || now - *lastUpdateTimepoint >= StatusUpdateInterval) {
            lastUpdateTimepoint = now;

            status.timestep = simulationFacade->getCurrentTimestep();
            status.realTime = simulationFacade->getRealTime();
            status.tps = simulationFacade->getTps();
            status.paused = !simulationFacade->isSimulationRunning();
            updateLiveOutput(liveOutput, status, simulationFacade);

            if (periodicUploader) {
                periodicUploader->process();
            }
        }
        _McpToolsFacade::get()->waitForToolCalls(StatusUpdateInterval);
    }
    if (periodicUploader) {
        periodicUploader->waitForPendingUpload();
    }
    liveOutput.close();

    simulationFacade->pauseSimulation();
}
