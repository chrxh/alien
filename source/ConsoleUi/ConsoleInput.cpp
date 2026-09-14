#include "ConsoleInput.h"

#include <atomic>
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

namespace
{
    std::atomic<bool> quitRequested = false;

#ifdef _WIN32
    BOOL WINAPI handleConsoleCtrlEvent(DWORD eventType)
    {
        // Closing the console window leaves too little time for saving, so it is left to the default handler
        if (eventType != CTRL_C_EVENT && eventType != CTRL_BREAK_EVENT) {
            return FALSE;
        }
        quitRequested.store(true);
        return TRUE;
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
    termios terminalAttributesBeforeBegin;
    int fileStatusFlagsBeforeBegin = 0;
    void (*interruptHandlerBeforeBegin)(int) = SIG_DFL;

    void handleInterrupt(int)
    {
        quitRequested.store(true);
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
}

void ConsoleInput::begin()
{
    quitRequested.store(false);

#ifdef _WIN32
    SetConsoleCtrlHandler(handleConsoleCtrlEvent, TRUE);
#else
    tcgetattr(STDIN_FILENO, &terminalAttributesBeforeBegin);
    auto attributes = terminalAttributesBeforeBegin;
    attributes.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &attributes);

    fileStatusFlagsBeforeBegin = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, fileStatusFlagsBeforeBegin | O_NONBLOCK);

    interruptHandlerBeforeBegin = std::signal(SIGINT, handleInterrupt);
#endif
}

void ConsoleInput::end()
{
#ifdef _WIN32
    SetConsoleCtrlHandler(handleConsoleCtrlEvent, FALSE);
#else
    std::signal(SIGINT, interruptHandlerBeforeBegin);

    tcsetattr(STDIN_FILENO, TCSANOW, &terminalAttributesBeforeBegin);
    fcntl(STDIN_FILENO, F_SETFL, fileStatusFlagsBeforeBegin);
#endif
}

std::optional<int> ConsoleInput::readPressedCharacter()
{
    auto keys = readPressedKeys();
    return keys.size() == 1 ? std::optional<int>(keys.front()) : std::nullopt;
}

bool ConsoleInput::isQuitRequested()
{
    return quitRequested.load();
}
