#include "ConsoleModeController.h"

#include <iostream>
#include <thread>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include <GLFW/glfw3.h>

#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

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

    void beginConsoleInput() {}

    void endConsoleInput() {}

    bool isEscapePressed()
    {
        auto result = false;
        while (_kbhit() != 0) {
            if (_getch() == EscapeKeyCode) {
                result = true;
            }
        }
        return result;
    }
#else
    termios terminalAttributesBeforeActivation;
    int fileStatusFlagsBeforeActivation = 0;

    void bringConsoleToFront() {}

    void beginConsoleInput()
    {
        tcgetattr(STDIN_FILENO, &terminalAttributesBeforeActivation);
        auto attributes = terminalAttributesBeforeActivation;
        attributes.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &attributes);

        fileStatusFlagsBeforeActivation = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, fileStatusFlagsBeforeActivation | O_NONBLOCK);
    }

    void endConsoleInput()
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &terminalAttributesBeforeActivation);
        fcntl(STDIN_FILENO, F_SETFL, fileStatusFlagsBeforeActivation);
    }

    bool isEscapePressed()
    {
        auto result = false;
        char input = 0;
        while (read(STDIN_FILENO, &input, 1) == 1) {
            if (input == EscapeKeyCode) {
                result = true;
            }
        }
        return result;
    }
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
    beginConsoleInput();

    auto window = WindowController::get().getWindowData().window;
    glfwIconifyWindow(window);
    glfwHideWindow(window);

    std::cout << std::endl
              << "Console mode: user interface and rendering are switched off." << std::endl
              << "Press ESC to return to the user interface." << std::endl
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
    printStatusLine();

    if (isEscapePressed()) {
        deactivate();
        return;
    }
    std::this_thread::sleep_for(PollInterval);
}

void ConsoleModeController::deactivate()
{
    endConsoleInput();
    std::cout << std::endl << "Returning to the user interface ..." << std::endl;

    auto simulationFacade = _SimulationFacade::get();
    simulationFacade->setSyncSimulationWithRendering(_stateBeforeActivation.syncSimulationWithRendering);
    simulationFacade->setTpsRestriction(_stateBeforeActivation.tpsRestriction);

    auto window = WindowController::get().getWindowData().window;
    glfwShowWindow(window);
    glfwRestoreWindow(window);
    glfwFocusWindow(window);

    _active = false;
}

void ConsoleModeController::printStatusLine()
{
    auto now = std::chrono::steady_clock::now();
    if (_lastPrintTimepoint && now - *_lastPrintTimepoint < PrintInterval) {
        return;
    }
    _lastPrintTimepoint = now;

    auto simulationFacade = _SimulationFacade::get();
    auto statusLine =
        "Time step: " + StringHelper::format(simulationFacade->getCurrentTimestep()) + "   TPS: " + StringHelper::format(simulationFacade->getTps(), 1);
    if (!simulationFacade->isSimulationRunning()) {
        statusLine += "   (paused)";
    }
    std::cout << "\r" << statusLine << "          " << std::flush;
}
