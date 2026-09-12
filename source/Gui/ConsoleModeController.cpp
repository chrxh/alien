#include "ConsoleModeController.h"

#include <atomic>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include <GLFW/glfw3.h>

#include <Base/Console.h>
#include <Base/Resources.h>

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

    std::atomic<bool> quitRequested = false;

#ifdef _WIN32
    void bringConsoleToFront()
    {
        if (auto consoleWindow = GetConsoleWindow()) {
            ShowWindow(consoleWindow, SW_RESTORE);
            SetForegroundWindow(consoleWindow);
        }
    }

    BOOL WINAPI handleConsoleCtrlEvent(DWORD eventType)
    {
        // Closing the console window leaves too little time for saving, so it is left to the default handler
        if (eventType != CTRL_C_EVENT && eventType != CTRL_BREAK_EVENT) {
            return FALSE;
        }
        quitRequested.store(true);
        return TRUE;
    }

    void beginConsoleInput()
    {
        SetConsoleCtrlHandler(handleConsoleCtrlEvent, TRUE);
    }

    void endConsoleInput()
    {
        SetConsoleCtrlHandler(handleConsoleCtrlEvent, FALSE);
    }

    std::vector<int> readPressedKeys()
    {
        std::vector<int> result;
        while (_kbhit() != 0) {
            result.emplace_back(_getch());
        }
        return result;
    }
#else
    termios terminalAttributesBeforeActivation;
    int fileStatusFlagsBeforeActivation = 0;

    void bringConsoleToFront() {}

    void handleInterrupt(int)
    {
        quitRequested.store(true);
    }

    void beginConsoleInput()
    {
        tcgetattr(STDIN_FILENO, &terminalAttributesBeforeActivation);
        auto attributes = terminalAttributesBeforeActivation;
        attributes.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &attributes);

        fileStatusFlagsBeforeActivation = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, fileStatusFlagsBeforeActivation | O_NONBLOCK);

        std::signal(SIGINT, handleInterrupt);
    }

    void endConsoleInput()
    {
        std::signal(SIGINT, SIG_DFL);

        tcsetattr(STDIN_FILENO, TCSANOW, &terminalAttributesBeforeActivation);
        fcntl(STDIN_FILENO, F_SETFL, fileStatusFlagsBeforeActivation);
    }

    std::vector<int> readPressedKeys()
    {
        std::vector<int> result;
        char input = 0;
        while (read(STDIN_FILENO, &input, 1) == 1) {
            result.emplace_back(input);
        }
        return result;
    }
#endif

    // Cursor and function keys deliver several bytes, so only a single byte counts as a pressed character
    std::optional<int> readPressedCharacter()
    {
        auto keys = readPressedKeys();
        return keys.size() == 1 ? std::optional<int>(keys.front()) : std::nullopt;
    }
}

void ConsoleModeController::activate()
{
    if (_active) {
        return;
    }
    _active = true;
    _lastPrintTimepoint.reset();
    quitRequested.store(false);

    auto simulationFacade = _SimulationFacade::get();
    _stateBeforeActivation = StateBeforeActivation{
        .syncSimulationWithRendering = simulationFacade->isSyncSimulationWithRendering(), .tpsRestriction = simulationFacade->getTpsRestriction()};

    simulationFacade->setSyncSimulationWithRendering(false);
    simulationFacade->setTpsRestriction(std::nullopt);

    bringConsoleToFront();
    beginConsoleInput();

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

    auto pressedCharacter = readPressedCharacter();
    if (quitRequested.load() || pressedCharacter == 'q' || pressedCharacter == 'Q') {
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
    endConsoleInput();
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
    std::cout << std::endl << "  " << ConsoleWidgets::createText("Returning to the user interface ...", ConsolePalette::Label) << std::endl;

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
    _status.duration = simulationFacade->getRealTime();
    _status.paused = !simulationFacade->isSimulationRunning();

    auto statistics = simulationFacade->getStatisticsEntry();
    _status.numCells = statistics.objectStatistics.numCellObjects;
    _status.numEnergyParticles = statistics.objectStatistics.numEnergyParticles;
    _status.numCreatures = 0;
    for (auto const& lineage : statistics.lineageEntries) {
        _status.numCreatures += lineage.numCreatures;
    }
    _status.numLineages = toUInt32(statistics.lineageEntries.size());
    _status.updateHistory();

    auto lines = ConsoleSimulationPanel::fitsIntoConsole() ? ConsoleSimulationPanel::create("console mode", _status) : std::vector<std::string>();
    _liveOutput.update(lines, ConsoleSimulationPanel::createPlainLine(_status));
}
