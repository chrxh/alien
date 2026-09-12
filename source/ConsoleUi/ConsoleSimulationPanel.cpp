#include "ConsoleSimulationPanel.h"

#include <Base/Definitions.h>
#include <Base/StringHelper.h>
#include "ConsoleWidgets.h"

namespace
{
    auto constexpr PanelWidth = 64;
    auto constexpr MaxPanelLines = 14;          // Of the progress variant, which is the taller one
    auto constexpr PlotWidth = PanelWidth - 4;  // Without the frame borders and the surrounding spaces
    size_t constexpr MaxHistorySize = PlotWidth;
    auto constexpr LabelWidth = 18;
    auto constexpr ValueWidth = 13;
    auto constexpr SecondLabelWidth = 12;
    auto constexpr SecondValueWidth = 11;
    auto constexpr SuffixGap = 2;

    std::string createRow(std::string const& label, std::string const& value, std::string const& suffix = std::string())
    {
        auto content = ConsoleWidgets::createField(label, value, LabelWidth, ValueWidth);
        if (!suffix.empty()) {
            content += std::string(SuffixGap, ' ') + suffix;
        }
        return ConsoleWidgets::createFrameRow(content, PanelWidth);
    }

    std::string createDoubleRow(std::string const& label, std::string const& value, std::string const& secondLabel, std::string const& secondValue)
    {
        auto content = ConsoleWidgets::createField(label, value, LabelWidth, ValueWidth) + "   "
            + ConsoleWidgets::createField(secondLabel, secondValue, SecondLabelWidth, SecondValueWidth);
        return ConsoleWidgets::createFrameRow(content, PanelWidth);
    }

    std::string createProgressRow(ConsoleSimulationStatus const& status)
    {
        auto fraction = *status.totalTimesteps != 0 ? toFloat(status.timestep) / toFloat(*status.totalTimesteps) : 0.0f;
        auto percentage = StringHelper::format(fraction * 100.0f, 1) + " %";
        auto barWidth = PanelWidth - 6 - Console::getVisibleLength(percentage);
        return ConsoleWidgets::createFrameRow(
            ConsoleWidgets::createProgressBar(fraction, barWidth) + "  " + ConsoleWidgets::createText(percentage, ConsolePalette::Value), PanelWidth);
    }

    std::chrono::milliseconds calcRemainingTime(ConsoleSimulationStatus const& status)
    {
        if (status.tps <= 0.0f || status.timestep >= *status.totalTimesteps) {
            return std::chrono::milliseconds(0);
        }
        return std::chrono::milliseconds(static_cast<int64_t>(toFloat(*status.totalTimesteps - status.timestep) / status.tps * 1000.0f));
    }
}

namespace
{
    void addToHistory(std::vector<float>& history, float value)
    {
        history.push_back(value);
        if (history.size() > MaxHistorySize) {
            history.erase(history.begin());
        }
    }
}

void ConsoleSimulationStatus::updateHistory()
{
    addToHistory(tpsHistory, tps);
}

bool ConsoleSimulationPanel::fitsIntoConsole()
{
    return Console::isRichOutput() && Console::getWidth() > PanelWidth && Console::getHeight() > MaxPanelLines;
}

std::vector<std::string> ConsoleSimulationPanel::create(std::string const& title, ConsoleSimulationStatus const& status)
{
    std::vector<std::string> result;
    result.push_back(ConsoleWidgets::createFrameTop(title, PanelWidth));

    if (status.totalTimesteps.has_value()) {
        result.push_back(createProgressRow(status));
        result.push_back(createRow(
            "time step",
            StringHelper::format(status.timestep),
            ConsoleWidgets::createText("/ " + StringHelper::format(*status.totalTimesteps), ConsolePalette::Label)));
        result.push_back(createRow("tps", StringHelper::format(status.tps, 1)));
        result.push_back(createDoubleRow("elapsed", StringHelper::format(status.duration), "remaining", StringHelper::format(calcRemainingTime(status))));
    } else {
        result.push_back(createRow(
            "time step", StringHelper::format(status.timestep), status.paused ? ConsoleWidgets::createText("paused", ConsolePalette::Warning) : std::string()));
        result.push_back(createRow("tps", StringHelper::format(status.tps, 1)));
        result.push_back(createRow("real time", StringHelper::format(status.duration)));
    }

    result.push_back(ConsoleWidgets::createFrameSeparator("world", PanelWidth));
    result.push_back(createRow("cells", StringHelper::format(status.numCells)));
    result.push_back(createRow("energy particles", StringHelper::format(status.numEnergyParticles)));
    result.push_back(createRow("creatures", StringHelper::format(status.numCreatures)));
    result.push_back(createRow("lineages", StringHelper::format(status.numLineages)));

    result.push_back(ConsoleWidgets::createFrameSeparator("tps", PanelWidth));
    for (auto const& plotLine : ConsoleWidgets::createPlot(status.tpsHistory, PlotWidth)) {
        result.push_back(ConsoleWidgets::createFrameRow(plotLine, PanelWidth));
    }
    result.push_back(ConsoleWidgets::createFrameBottom(PanelWidth));
    return result;
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
