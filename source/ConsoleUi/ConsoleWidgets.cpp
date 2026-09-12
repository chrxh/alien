#include "ConsoleWidgets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

namespace
{
    auto constexpr FrameHorizontal = "\xe2\x94\x80";
    auto constexpr FrameVertical = "\xe2\x94\x82";
    auto constexpr FrameTopLeft = "\xe2\x94\x8c";
    auto constexpr FrameTopRight = "\xe2\x94\x90";
    auto constexpr FrameBottomLeft = "\xe2\x94\x94";
    auto constexpr FrameBottomRight = "\xe2\x94\x98";
    auto constexpr FrameLeftJoint = "\xe2\x94\x9c";
    auto constexpr FrameRightJoint = "\xe2\x94\xa4";

    auto constexpr ShadeLight = "\xe2\x96\x91";
    auto constexpr ShadeFull = "\xe2\x96\x88";

    auto constexpr BannerWidth = 37;
    auto constexpr BannerIndent = 2;

    auto constexpr WordmarkWidth = 35;
    std::array<char const*, 5> const WordmarkArt = {
        " ####   ##      ##  ######  ##   ##",
        "##  ##  ##      ##  ##      ###  ##",
        "######  ##      ##  #####   ## # ##",
        "##  ##  ##      ##  ##      ##  ###",
        "##  ##  ######  ##  ######  ##   ##"};

    std::string repeat(char const* glyph, int count)
    {
        std::string result;
        while (count > 0) {
            result += glyph;
            --count;
        }
        return result;
    }

    ConsoleColor getBannerColor(float fraction)
    {
        if (fraction < 0.5f) {
            return Console::blend(ConsolePalette::LogoBlue, ConsolePalette::LogoMagenta, fraction * 2.0f);
        }
        return Console::blend(ConsolePalette::LogoMagenta, ConsolePalette::LogoRed, (fraction - 0.5f) * 2.0f);
    }

    std::vector<std::string> renderWordmark(int artOffset)
    {
        std::vector<std::string> result;
        for (auto const& row : WordmarkArt) {
            auto line = std::string(BannerIndent + artOffset, ' ');
            auto column = artOffset;
            for (auto const& pixel : std::string_view(row)) {
                if (pixel == '#') {
                    line += Console::foreground(getBannerColor(static_cast<float>(column) / static_cast<float>(BannerWidth - 1))) + ShadeFull;
                } else {
                    line += ' ';
                }
                ++column;
            }
            result.push_back(line + Console::reset());
        }
        return result;
    }

}

std::vector<std::string> ConsoleWidgets::createBanner(std::string const& subtitle)
{
    if (!Console::isRichOutput()) {
        return {"ALIEN - " + subtitle};
    }
    auto result = renderWordmark((BannerWidth - WordmarkWidth) / 2);
    result.emplace_back();

    result.push_back(std::string(BannerIndent + 1, ' ') + createText(subtitle, ConsolePalette::Label));
    return result;
}

namespace
{
    std::string createFrameHeader(char const* leftCorner, char const* rightCorner, std::string const& title, int width)
    {
        auto frameColor = Console::foreground(ConsolePalette::Frame);
        auto result = frameColor + leftCorner + FrameHorizontal;
        auto column = ConsoleWidgets::FrameContentOffset;
        if (!title.empty()) {
            result += " " + ConsoleWidgets::createText(title, ConsolePalette::Accent) + frameColor + " ";
            column += Console::getVisibleLength(title) + 2;
        }
        return result + repeat(FrameHorizontal, std::max(0, width - 1 - column)) + rightCorner + Console::reset();
    }
}

std::string ConsoleWidgets::createFrameTop(std::string const& title, int width)
{
    if (!Console::isRichOutput()) {
        return title;
    }
    return createFrameHeader(FrameTopLeft, FrameTopRight, title, width);
}

std::string ConsoleWidgets::createFrameSeparator(std::string const& title, int width)
{
    if (!Console::isRichOutput()) {
        return title;
    }
    return createFrameHeader(FrameLeftJoint, FrameRightJoint, title, width);
}

std::string ConsoleWidgets::createFrameBottom(int width)
{
    if (!Console::isRichOutput()) {
        return std::string();
    }
    return Console::foreground(ConsolePalette::Frame) + FrameBottomLeft + repeat(FrameHorizontal, std::max(0, width - 2)) + FrameBottomRight + Console::reset();
}

std::string ConsoleWidgets::createFrameRow(std::string const& content, int width)
{
    if (!Console::isRichOutput()) {
        return content;
    }
    auto border = Console::foreground(ConsolePalette::Frame) + FrameVertical + Console::reset();
    auto padding = std::max(0, width - 2 * FrameContentOffset - Console::getVisibleLength(content));
    return border + " " + content + std::string(padding, ' ') + " " + border;
}

std::string ConsoleWidgets::createProgressBar(float fraction, int width)
{
    auto filledWidth = static_cast<int>(std::round(std::clamp(fraction, 0.0f, 1.0f) * static_cast<float>(width)));
    if (!Console::isRichOutput()) {
        return std::string(filledWidth, '#') + std::string(std::max(0, width - filledWidth), '.');
    }
    std::string result;
    auto column = 0;
    while (column < filledWidth) {
        result += Console::foreground(getBannerColor(width > 1 ? static_cast<float>(column) / static_cast<float>(width - 1) : 0.0f));
        result += ShadeFull;
        ++column;
    }
    return result + Console::foreground(ConsolePalette::Frame) + repeat(ShadeLight, std::max(0, width - filledWidth)) + Console::reset();
}

std::string ConsoleWidgets::createField(std::string const& label, std::string const& value, int labelWidth, int valueWidth)
{
    auto labelPadding = std::max(0, labelWidth - Console::getVisibleLength(label));
    auto valuePadding = std::max(0, valueWidth - Console::getVisibleLength(value));
    return createText(label, ConsolePalette::Label) + std::string(labelPadding, ' ') + std::string(valuePadding, ' ')
        + createText(value, ConsolePalette::Value);
}

std::string ConsoleWidgets::createText(std::string const& text, ConsoleColor const& color)
{
    return Console::foreground(color) + text + Console::reset();
}
