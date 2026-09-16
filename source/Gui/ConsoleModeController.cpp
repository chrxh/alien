#include "ConsoleModeController.h"

#include <iostream>
#include <optional>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Base/Console.h>
#include <Base/Resources.h>

#include <ConsoleUi/ConsoleInput.h>
#include <ConsoleUi/ConsoleWidgets.h>

#include <EngineInterface/SimulationFacade.h>

#include "AutosaveController.h"
#include "BrowserController.h"
#include "MainLoopController.h"
#include "WindowController.h"

namespace
{
    auto constexpr PollInterval = std::chrono::milliseconds(100);
    auto constexpr PrintInterval = std::chrono::milliseconds(200);
    auto constexpr EscapeKeyCode = 27;

#ifdef _WIN32
    void bringConsoleToFront()
    {
        if (auto consoleWindow = GetConsoleWindow()) {
            ShowWindow(consoleWindow, SW_RESTORE);
            SetForegroundWindow(consoleWindow);
        }
    }
#else
    void bringConsoleToFront() {}
#endif
}

void ConsoleModeController::activate()
{
    if (_active) {
        return;
    }
    _active = true;
    _lastPrintTimepoint.reset();

    auto simulationFacade = _SimulationFacade::get();
    _stateBeforeActivation = StateBeforeActivation{
        .syncSimulationWithRendering = simulationFacade->isSyncSimulationWithRendering(), .tpsRestriction = simulationFacade->getTpsRestriction()};

    simulationFacade->setSyncSimulationWithRendering(false);
    simulationFacade->setTpsRestriction(std::nullopt);

    bringConsoleToFront();
    ConsoleInput::begin();

    auto window = WindowController::get().getWindowData().window;
    glfwIconifyWindow(window);
    glfwHideWindow(window);

    Console::init(false);
    std::cout << Console::clearScreen() << std::endl;
    for (auto const& line : ConsoleWidgets::createBanner("artificial life environment  \xc2\xb7  v" + Const::ProgramVersion + "  \xc2\xb7  console mode")) {
        std::cout << line << std::endl;
    }
    std::cout << std::endl
              << "  " << ConsoleWidgets::createText("User interface and rendering are switched off.", ConsolePalette::Label) << std::endl
              << "  " << ConsoleWidgets::createText("Press ESC to return to the user interface.", ConsolePalette::Label) << std::endl
              << "  " << ConsoleWidgets::createText("Press Q to quit, which saves on exit like the user interface does.", ConsolePalette::Label) << std::endl
              << std::endl;
}

bool ConsoleModeController::isActive() const
{
    return _active;
}

void ConsoleModeController::process()
{
    if (!_active) {
        return;
    }
    AutosaveController::get().process();
    BrowserController::get().process();
    printPersistedSavepoint();
    printStatusLine();

    auto pressedCharacter = ConsoleInput::readPressedCharacter();
    if (ConsoleInput::isQuitRequested() || pressedCharacter == 'q' || pressedCharacter == 'Q') {
        quit();
        return;
    }
    if (pressedCharacter == EscapeKeyCode) {
        deactivate();
        return;
    }
    std::this_thread::sleep_for(PollInterval);
}

void ConsoleModeController::quit()
{
    leaveConsoleMode();
    auto message = MainLoopController::get().isSaveOnExit() ? "Saving on exit ..." : "Quitting without saving ...";
    std::cout << std::endl << "  " << ConsoleWidgets::createText(message, ConsolePalette::Label) << std::endl;

    // The main loop runs its regular exit sequence again, but leaves the window hidden
    MainLoopController::get().scheduleClosing();
}

void ConsoleModeController::leaveConsoleMode()
{
    ConsoleInput::end();
    _liveOutput.close();
    _status = ConsoleSimulationStatus();

    auto simulationFacade = _SimulationFacade::get();
    simulationFacade->setSyncSimulationWithRendering(_stateBeforeActivation.syncSimulationWithRendering);
    simulationFacade->setTpsRestriction(_stateBeforeActivation.tpsRestriction);

    _active = false;
}

void ConsoleModeController::deactivate()
{
    leaveConsoleMode();
    std::cout << Console::clearScreen() << std::flush;

    auto window = WindowController::get().getWindowData().window;
    glfwShowWindow(window);
    glfwRestoreWindow(window);
    glfwFocusWindow(window);
}

void ConsoleModeController::printPersistedSavepoint()
{
    auto savepoint = AutosaveController::get().getPersistedSavepoint();
    if (!savepoint.has_value()) {
        return;
    }
    _liveOutput.printMessage(
        "  " + ConsoleWidgets::createText("Save point created: ", ConsolePalette::Success)
        + ConsoleWidgets::createText(savepoint.value()->filename.string(), ConsolePalette::Value));
    _lastPrintTimepoint.reset();
}

void ConsoleModeController::printStatusLine()
{
    auto now = std::chrono::steady_clock::now();
    if (_lastPrintTimepoint && now - *_lastPrintTimepoint < PrintInterval) {
        return;
    }
    _lastPrintTimepoint = now;

    auto simulationFacade = _SimulationFacade::get();
    _status.timestep = simulationFacade->getCurrentTimestep();
    _status.tps = simulationFacade->getTps();
    _status.realTime = simulationFacade->getRealTime();
    _status.paused = !simulationFacade->isSimulationRunning();

    auto statistics = simulationFacade->getStatisticsEntry();
    _status.numCells = statistics.objectStatistics.numCellObjects;
    _status.numCreatures = 0;
    for (auto const& lineage : statistics.lineageEntries) {
        _status.numCreatures += lineage.numCreatures;
    }
    _status.numLineages = toUInt32(statistics.lineageEntries.size());

    auto lines = ConsoleSimulationPanel::fitsIntoConsole() ? ConsoleSimulationPanel::create(_status) : std::vector<std::string>();
    _liveOutput.update(lines, ConsoleSimulationPanel::createPlainLine(_status));
}
