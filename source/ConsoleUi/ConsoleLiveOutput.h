#pragma once

#include <string>
#include <vector>

// Prints a block of lines that is redrawn in place on every update
class ConsoleLiveOutput
{
public:
    ~ConsoleLiveOutput();

    // Falls back to the single line summary if the panel lines are empty or rich output is not available
    void update(std::vector<std::string> const& lines, std::string const& plainLine);

    void printMessage(std::string const& message);
    void close();

private:
    std::string createEraseSequence();

    std::vector<std::string> _printedLines;
    int _plainLineLength = 0;
    bool _cursorHidden = false;
};
