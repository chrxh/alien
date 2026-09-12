#include "ConsoleSimulationPanel.h"

#include <algorithm>
#include <ranges>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>
#include "ConsoleWidgets.h"

namespace
{
    constexpr int getContentWidth(int cardWidth)
    {
        return cardWidth - 2 * ConsoleWidgets::FrameContentOffset;
    }

    auto constexpr TimeCardWidth = 33;
    auto constexpr ProgressTimeCardWidth = 42;  // Wider, because the time step row also shows the total number
    auto constexpr WorldCardWidth = 33;
    auto constexpr CardGap = 2;
    auto constexpr PanelWidth = ProgressTimeCardWidth + CardGap + WorldCardWidth;
    auto constexpr MaxPanelLines = 7;  // Of the progress variant, which is the taller one
    auto constexpr TimeLabelWidth = 10;
    auto constexpr WorldLabelWidth = 17;
    auto constexpr WorldValueWidth = getContentWidth(WorldCardWidth) - WorldLabelWidth;
    auto constexpr SuffixGap = 2;
}

bool ConsoleSimulationPanel::fitsIntoConsole()
{
    return Console::isRichOutput() && Console::getWidth() > PanelWidth && Console::getHeight() > MaxPanelLines;
}

namespace
{
    int getTimeCardWidth(ConsoleSimulationStatus const& status)
    {
        return status.totalTimesteps.has_value() ? ProgressTimeCardWidth : TimeCardWidth;
    }

    std::string createTimeRow(int width, std::string const& label, std::string const& value, std::string const& suffix = std::string())
    {
        auto suffixWidth = suffix.empty() ? 0 : SuffixGap + Console::getVisibleLength(suffix);
        auto content = ConsoleWidgets::createField(label, value, TimeLabelWidth, getContentWidth(width) - TimeLabelWidth - suffixWidth);
        if (!suffix.empty()) {
            content += std::string(SuffixGap, ' ') + suffix;
        }
        return ConsoleWidgets::createFrameRow(content, width);
    }

    std::string createProgressRow(ConsoleSimulationStatus const& status)
    {
        auto fraction = *status.totalTimesteps != 0 ? toFloat(status.timestep) / toFloat(*status.totalTimesteps) : 0.0f;
        auto percentage = StringHelper::format(fraction * 100.0f, 1) + " %";
        auto barWidth = ProgressTimeCardWidth - 6 - Console::getVisibleLength(percentage);
        return ConsoleWidgets::createFrameRow(
            ConsoleWidgets::createProgressBar(fraction, barWidth) + "  " + ConsoleWidgets::createText(percentage, ConsolePalette::Value),
            ProgressTimeCardWidth);
    }

    std::chrono::milliseconds calcRemainingTime(ConsoleSimulationStatus const& status)
    {
        if (status.tps <= 0.0f || status.timestep >= *status.totalTimesteps) {
            return std::chrono::milliseconds(0);
        }
        return std::chrono::milliseconds(static_cast<int64_t>(toFloat(*status.totalTimesteps - status.timestep) / status.tps * 1000.0f));
    }

    std::vector<std::string> createTimeCard(ConsoleSimulationStatus const& status)
    {
        auto width = getTimeCardWidth(status);

        std::vector<std::string> result;
        result.push_back(ConsoleWidgets::createFrameTop("time", width));
        if (status.totalTimesteps.has_value()) {
            result.push_back(createProgressRow(status));
            result.push_back(createTimeRow(
                width,
                "time step",
                StringHelper::format(status.timestep),
                ConsoleWidgets::createText("/ " + StringHelper::format(*status.totalTimesteps), ConsolePalette::Label)));
            result.push_back(createTimeRow(width, "tps", StringHelper::format(status.tps, 1)));
            result.push_back(createTimeRow(width, "elapsed", StringHelper::format(status.duration)));
            result.push_back(createTimeRow(width, "remaining", StringHelper::format(calcRemainingTime(status))));
        } else {
            result.push_back(createTimeRow(
                width,
                "time step",
                StringHelper::format(status.timestep),
                status.paused ? ConsoleWidgets::createText("paused", ConsolePalette::Warning) : std::string()));
            result.push_back(createTimeRow(width, "tps", StringHelper::format(status.tps, 1)));
            result.push_back(createTimeRow(width, "real time", StringHelper::format(status.duration)));
        }
        result.push_back(ConsoleWidgets::createFrameBottom(width));
        return result;
    }

    std::vector<std::string> createWorldCard(ConsoleSimulationStatus const& status)
    {
        auto createRow = [](std::string const& label, std::string const& value) {
            return ConsoleWidgets::createFrameRow(ConsoleWidgets::createField(label, value, WorldLabelWidth, WorldValueWidth), WorldCardWidth);
        };
        return {
            ConsoleWidgets::createFrameTop("world", WorldCardWidth),
            createRow("cells", StringHelper::format(status.numCells)),
            createRow("creatures", StringHelper::format(status.numCreatures)),
            createRow("lineages", StringHelper::format(status.numLineages)),
            ConsoleWidgets::createFrameBottom(WorldCardWidth)};
    }

    std::vector<std::string> joinSideBySide(std::vector<std::string> leftCard, int leftCardWidth, std::vector<std::string> rightCard)
    {
        auto numLines = std::max(leftCard.size(), rightCard.size());
        leftCard.resize(numLines, std::string(leftCardWidth, ' '));
        rightCard.resize(numLines);

        std::vector<std::string> result;
        for (auto const& [leftLine, rightLine] : std::views::zip(leftCard, rightCard)) {
            result.push_back(rightLine.empty() ? leftLine : leftLine + std::string(CardGap, ' ') + rightLine);
        }
        return result;
    }
}

std::vector<std::string> ConsoleSimulationPanel::create(ConsoleSimulationStatus const& status)
{
    return joinSideBySide(createTimeCard(status), getTimeCardWidth(status), createWorldCard(status));
}

std::string ConsoleSimulationPanel::createPlainLine(ConsoleSimulationStatus const& status)
{
    auto result = "Time step: " + StringHelper::format(status.timestep);
    if (status.totalTimesteps.has_value()) {
        result += " / " + StringHelper::format(*status.totalTimesteps);
    }
    result += "   TPS: " + StringHelper::format(status.tps, 1) + "   Cells: " + StringHelper::format(status.numCells)
        + "   Creatures: " + StringHelper::format(status.numCreatures) + "   Lineages: " + StringHelper::format(status.numLineages);
    if (status.paused) {
        result += "   (paused)";
    }
    return result;
}
