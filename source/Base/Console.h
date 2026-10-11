#pragma once

#include <cstdint>
#include <string>

struct ConsoleColor
{
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

// Terminal capabilities and ANSI escape sequences. Without rich output all returned sequences are empty.
class Console
{
public:
    static void init(bool plainOutput);

    static bool isRichOutput();
    static int getWidth();
    static int getHeight();

    static std::string foreground(ConsoleColor const& color);
    static std::string reset();
    static std::string bold();
    static std::string dim();
    static std::string hideCursor();
    static std::string showCursor();

    static std::string moveUpAndErase(int numLines);

    static std::string moveUp(int numLines);
    static std::string moveToNextLine();
    static std::string eraseToEndOfLine();

    static std::string clearScreen();

    static ConsoleColor blend(ConsoleColor const& first, ConsoleColor const& second, float fraction);

    // Length without escape sequences, counting a multi-byte utf-8 character as one
    static int getVisibleLength(std::string const& text);
};
