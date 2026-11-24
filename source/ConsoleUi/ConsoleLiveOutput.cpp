#include "ConsoleLiveOutput.h"

#include <iostream>
#include <ranges>

#include <Base/Console.h>

namespace
{
    auto constexpr LinesBetweenMessageAndPanel = 2;
}

ConsoleLiveOutput::~ConsoleLiveOutput()
{
    close();
}

void ConsoleLiveOutput::update(std::vector<std::string> const& lines, std::string const& plainLine)
{
    if (!Console::isRichOutput() || lines.empty()) {
        if (plainLine.empty() || plainLine == _printedPlainLine) {
            return;
        }
        auto output = createEraseSequence() + plainLine;
        _printedPlainLine = plainLine;
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
    auto panelLines = _printedLines;
    auto plainLine = _printedPlainLine;

    auto output = createEraseSequence();
    if (!_lastMessage.empty()) {
        output += Console::moveUp(1);
    }
    std::cout << output << message << Console::eraseToEndOfLine() << std::endl;
    if (Console::isRichOutput()) {

        // The trailing blank line separates the message from the panel that is redrawn below
        std::cout << std::endl;
    }
    _lastMessage = message;

    update(panelLines, plainLine);
}

void ConsoleLiveOutput::appendToMessage(std::string const& message, std::string const& text)
{
    if (!Console::isRichOutput() || _lastMessage != message) {
        printMessage(message + text);
        return;
    }
    _lastMessage += text;

    auto panelLines = _printedLines;
    std::cout << createEraseSequence() << Console::moveUp(LinesBetweenMessageAndPanel) << _lastMessage << Console::eraseToEndOfLine()
              << Console::moveToNextLine() << Console::moveToNextLine() << std::flush;
    update(panelLines, std::string());
}

void ConsoleLiveOutput::close()
{
    if (!_printedPlainLine.empty()) {
        std::cout << std::endl;
        _printedPlainLine.clear();
    }
    if (_cursorHidden) {
        std::cout << Console::showCursor() << std::flush;
        _cursorHidden = false;
    }
    _printedLines.clear();
    _lastMessage.clear();
}

std::string ConsoleLiveOutput::createEraseSequence()
{
    auto result = std::string();
    if (!_printedLines.empty()) {
        result = Console::moveUpAndErase(static_cast<int>(_printedLines.size()));
        _printedLines.clear();
    }
    if (!_printedPlainLine.empty()) {
        result += "\r" + std::string(Console::getVisibleLength(_printedPlainLine), ' ') + "\r";
        _printedPlainLine.clear();
    }
    return result;
}
