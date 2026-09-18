#pragma once

#include <string>

class ConsoleOutput
{
public:
    static void installInterruptHandler();

    static void printBanner();
    static void printStep(std::string const& label, std::string const& value, std::string const& detail = std::string());
    static void printHint(std::string const& text);
    static void printBlock(std::string const& text);
    static void printError(std::string const& message);
    static void printBlankLine();
};
