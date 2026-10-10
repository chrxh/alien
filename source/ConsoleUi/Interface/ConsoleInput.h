#pragma once

#include <optional>

class ConsoleInput
{
public:
    static void begin();
    static void end();

    // Cursor and function keys deliver several bytes, so only a single byte counts as a pressed character
    static std::optional<int> readPressedCharacter();

    static bool isQuitRequested();
};
