#include "ConsoleLiveOutput.h"

#include <iostream>
#include <ranges>

#include <Base/Console.h>

ConsoleLiveOutput::~ConsoleLiveOutput()
{
    close();
}

void ConsoleLiveOutput::update(std::vector<std::string> const& lines, std::string const& plainLine)
{
    if (!Console::isRichOutput() || lines.empty()) {
        if (plainLine.empty()) {
            return;
        }
        auto output = createEraseSequence() + plainLine;
        _plainLineLength = Console::getVisibleLength(plainLine);
        std::cout << output << std::flush;
        return;
    }

    std::string output;
    if (!_cursorHidden) {
        output += Console::hideCursor();
        _cursorHidden = true;
    }

    // Overwriting in place avoids the flicker of erasing the panel before redrawing it
    if (_printedLines.size() != lines.size()) {
        output += createEraseSequence();
        for (auto const& line : lines) {
            output += line + Console::eraseToEndOfLine() + "\n";
        }
    } else {
        output += Console::moveUp(static_cast<int>(lines.size()));
        for (auto const& [printedLine, line] : std::views::zip(_printedLines, lines)) {
            if (printedLine != line) {
                output += line + Console::eraseToEndOfLine();
            }
            output += Console::moveToNextLine();
        }
    }
    _printedLines = lines;
    std::cout << output << std::flush;
}

void ConsoleLiveOutput::printMessage(std::string const& message)
{
    std::cout << createEraseSequence() << message << std::endl;
}

void ConsoleLiveOutput::close()
{
    if (_plainLineLength > 0) {
        std::cout << std::endl;
        _plainLineLength = 0;
    }
    if (_cursorHidden) {
        std::cout << Console::showCursor() << std::flush;
        _cursorHidden = false;
    }
    _printedLines.clear();
}

std::string ConsoleLiveOutput::createEraseSequence()
{
    auto result = std::string();
    if (!_printedLines.empty()) {
        result = Console::moveUpAndErase(static_cast<int>(_printedLines.size()));
        _printedLines.clear();
    }
    if (_plainLineLength > 0) {
        result += "\r" + std::string(_plainLineLength, ' ') + "\r";
        _plainLineLength = 0;
    }
    return result;
}
