#include "ConsoleOutput.h"

#include <algorithm>
#include <iostream>
#include <csignal>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <Base/Console.h>
#include <Base/Resources.h>

#include <ConsoleUi/ConsoleWidgets.h>

namespace
{
    auto constexpr StepLabelWidth = 10;
    auto constexpr CheckMark = "\xe2\x9c\x94";

    // Written directly, because a signal handler must not allocate or take the stdio lock
    void restoreCursorOnInterrupt(int signalNumber)
    {
        char constexpr ShowCursor[] = "\x1b[?25h";
        if (Console::isRichOutput()) {
#ifdef _WIN32
            _write(_fileno(stdout), ShowCursor, sizeof(ShowCursor) - 1);
#else
            auto const written = write(STDOUT_FILENO, ShowCursor, sizeof(ShowCursor) - 1);
            static_cast<void>(written);
#endif
        }
        std::signal(signalNumber, SIG_DFL);
        std::raise(signalNumber);
    }
}

void ConsoleOutput::installInterruptHandler()
{
    std::signal(SIGINT, restoreCursorOnInterrupt);
}

void ConsoleOutput::clearScreen()
{
    std::cout << Console::clearScreen() << std::flush;
}

void ConsoleOutput::printBanner()
{
    std::cout << std::endl;
    for (auto const& line : ConsoleWidgets::createBanner("artificial life environment  \xc2\xb7  v" + Const::ProgramVersion + "  \xc2\xb7  command line")) {
        std::cout << line << std::endl;
    }
    std::cout << std::endl;
}

void ConsoleOutput::printStep(std::string const& label, std::string const& value, std::string const& detail)
{
    if (!Console::isRichOutput()) {
        std::cout << label << ": " << value << (detail.empty() ? "" : " (" + detail + ")") << std::endl;
        return;
    }
    auto padding = std::max(0, StepLabelWidth - Console::getVisibleLength(label));
    std::cout << "  " << ConsoleWidgets::createText(CheckMark, ConsolePalette::Success) << " " << ConsoleWidgets::createText(label, ConsolePalette::Label)
              << std::string(padding, ' ') << ConsoleWidgets::createText(value, ConsolePalette::Value);
    if (!detail.empty()) {
        std::cout << "  " << ConsoleWidgets::createText(detail, ConsolePalette::Label);
    }
    std::cout << std::endl;
}

void ConsoleOutput::printHint(std::string const& text)
{
    std::cout << std::endl << "  " << ConsoleWidgets::createText(text, ConsolePalette::Label) << std::endl;
}

void ConsoleOutput::printBlock(std::string const& text)
{
    std::cout << std::endl << text << std::endl;
}

void ConsoleOutput::printError(std::string const& message)
{
    std::cout << std::endl << "  " << ConsoleWidgets::createText(message, ConsolePalette::Error) << std::endl;
}

void ConsoleOutput::printBlankLine()
{
    std::cout << std::endl;
}
