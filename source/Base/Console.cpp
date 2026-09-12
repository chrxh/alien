#include "Console.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

#ifdef _WIN32
#include <io.h>

#include <Windows.h>
#else
#include <cstdlib>

#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace
{
    auto constexpr ControlSequenceIntroducer = "\x1b[";
    auto constexpr FallbackWidth = 80;
    auto constexpr MinWidth = 40;
    auto constexpr MaxWidth = 200;
    auto constexpr FallbackHeight = 25;

    bool richOutput = false;

    bool isOutputToTerminal()
    {
#ifdef _WIN32
        return _isatty(_fileno(stdout)) != 0;
#else
        return isatty(fileno(stdout)) != 0;
#endif
    }

    bool isEnvironmentVariableSet(char const* name)
    {
#ifdef _WIN32
        return GetEnvironmentVariableA(name, nullptr, 0) != 0;
#else
        return std::getenv(name) != nullptr;
#endif
    }

    void setUtf8OutputCodePage()
    {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
    }

    bool enableEscapeSequences()
    {
#ifdef _WIN32
        auto handle = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (handle == INVALID_HANDLE_VALUE || GetConsoleMode(handle, &mode) == 0) {
            return false;
        }
        return SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
        return true;
#endif
    }
}

void Console::init(bool plainOutput)
{
    setUtf8OutputCodePage();
    if (plainOutput || isEnvironmentVariableSet("NO_COLOR")) {
        richOutput = false;
        return;
    }
    // Must not be short-circuited by CLICOLOR_FORCE, which only decides about redirected output
    auto escapeSequencesEnabled = enableEscapeSequences();
    richOutput = isEnvironmentVariableSet("CLICOLOR_FORCE") || (isOutputToTerminal() && escapeSequencesEnabled);
}

bool Console::isRichOutput()
{
    return richOutput;
}

int Console::getWidth()
{
    auto result = FallbackWidth;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo) != 0) {
        result = bufferInfo.srWindow.Right - bufferInfo.srWindow.Left + 1;
    }
#else
    winsize windowSize{};
    if (ioctl(fileno(stdout), TIOCGWINSZ, &windowSize) == 0 && windowSize.ws_col > 0) {
        result = windowSize.ws_col;
    }
#endif
    return std::clamp(result, MinWidth, MaxWidth);
}

int Console::getHeight()
{
    auto result = FallbackHeight;
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo) != 0) {
        result = bufferInfo.srWindow.Bottom - bufferInfo.srWindow.Top + 1;
    }
#else
    winsize windowSize{};
    if (ioctl(fileno(stdout), TIOCGWINSZ, &windowSize) == 0 && windowSize.ws_row > 0) {
        result = windowSize.ws_row;
    }
#endif
    return std::max(result, 1);
}

std::string Console::foreground(ConsoleColor const& color)
{
    if (!richOutput) {
        return std::string();
    }
    return std::string(ControlSequenceIntroducer) + "38;2;" + std::to_string(color.r) + ";" + std::to_string(color.g) + ";" + std::to_string(color.b) + "m";
}

std::string Console::reset()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "0m" : std::string();
}

std::string Console::bold()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "1m" : std::string();
}

std::string Console::dim()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "2m" : std::string();
}

std::string Console::hideCursor()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "?25l" : std::string();
}

std::string Console::showCursor()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "?25h" : std::string();
}

std::string Console::moveUpAndErase(int numLines)
{
    if (!richOutput) {
        return std::string();
    }
    return moveUp(numLines) + ControlSequenceIntroducer + "0J";
}

std::string Console::moveUp(int numLines)
{
    if (!richOutput) {
        return std::string();
    }
    if (numLines <= 0) {
        return "\r";
    }
    return std::string(ControlSequenceIntroducer) + std::to_string(numLines) + "F";
}

std::string Console::moveToNextLine()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "1E" : std::string();
}

std::string Console::eraseToEndOfLine()
{
    return richOutput ? std::string(ControlSequenceIntroducer) + "0K" : std::string();
}

std::string Console::clearScreen()
{
    if (!richOutput) {
        return std::string();
    }
    return std::string(ControlSequenceIntroducer) + "2J" + ControlSequenceIntroducer + "3J" + ControlSequenceIntroducer + "H";
}

ConsoleColor Console::blend(ConsoleColor const& first, ConsoleColor const& second, float fraction)
{
    auto mix = [fraction](uint8_t from, uint8_t to) {
        return static_cast<uint8_t>(std::clamp(static_cast<float>(from) + (static_cast<float>(to) - static_cast<float>(from)) * fraction, 0.0f, 255.0f));
    };
    return ConsoleColor{.r = mix(first.r, second.r), .g = mix(first.g, second.g), .b = mix(first.b, second.b)};
}

int Console::getVisibleLength(std::string const& text)
{
    auto result = 0;
    auto insideEscapeSequence = false;
    for (auto const& character : text) {
        if (insideEscapeSequence) {
            if (std::isalpha(static_cast<unsigned char>(character)) != 0) {
                insideEscapeSequence = false;
            }
            continue;
        }
        if (character == '\x1b') {
            insideEscapeSequence = true;
            continue;
        }
        if ((static_cast<unsigned char>(character) & 0xc0) != 0x80) {
            ++result;
        }
    }
    return result;
}
