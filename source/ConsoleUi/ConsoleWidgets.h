#pragma once

#include <string>
#include <vector>

#include <Base/Console.h>

namespace ConsolePalette
{
    ConsoleColor constexpr LogoBlue = {60, 105, 255};
    ConsoleColor constexpr LogoMagenta = {235, 75, 235};
    ConsoleColor constexpr LogoRed = {255, 65, 65};
    ConsoleColor constexpr Frame = {85, 95, 115};
    ConsoleColor constexpr Label = {135, 145, 165};
    ConsoleColor constexpr Value = {225, 230, 240};
    ConsoleColor constexpr Accent = {90, 200, 255};
    ConsoleColor constexpr Success = {120, 220, 130};
    ConsoleColor constexpr Warning = {255, 180, 60};
    ConsoleColor constexpr Error = {255, 95, 95};
}

class ConsoleWidgets
{
public:
    static std::vector<std::string> createBanner(std::string const& subtitle);

    static std::string createFrameTop(std::string const& title, int width);
    static std::string createFrameSeparator(std::string const& title, int width);
    static std::string createFrameBottom(int width);
    static std::string createFrameRow(std::string const& content, int width);

    static std::string createProgressBar(float fraction, int width);

    static std::string createField(std::string const& label, std::string const& value, int labelWidth, int valueWidth);
    static std::string createText(std::string const& text, ConsoleColor const& color);
};
