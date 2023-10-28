#include "StatisticsWindow.h"

#include <fstream>

#include <ImFileDialog.h>
#include <imgui.h>
#include <implot.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/ExportService.h"

#include "StyleRepository.h"
#include "AlienImGui.h"
#include "GenericFileDialogs.h"
#include "MessageDialog.h"

namespace
{
    auto const RightColumnWidth = 175.0f;
    auto const RightColumnWidthTable = 150.0f;
    auto const PlotMaxHeight = 160.0f;
    auto const PlotMinHeight = 80.0f;
}

_StatisticsWindow::_StatisticsWindow(SimulationController const& simController)
    : _AlienWindow("Statistics", "windows.statistics", false)
    , _simController(simController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("windows.statistics.starting path", path.string());
    _maximize = GlobalSettings::getInstance().getBoolState("windows.statistics.maximized", _maximize);
    _live = GlobalSettings::getInstance().getBoolState("windows.statistics.live", _live);
    _plotType = GlobalSettings::getInstance().getIntState("windows.statistics.plot type", _plotType);
}

_StatisticsWindow::~_StatisticsWindow()
{
    GlobalSettings::getInstance().setStringState("windows.statistics.starting path", _startingPath);
    GlobalSettings::getInstance().setBoolState("windows.statistics.maximized", _maximize);
    GlobalSettings::getInstance().setBoolState("windows.statistics.live", _live);
    GlobalSettings::getInstance().setIntState("windows.statistics.plot type", _plotType);
}

void _StatisticsWindow::reset()
{
    _liveStatistics = TimelineLiveStatistics();
    _longtermStatistics = TimelineLongtermStatistics();
}

void _StatisticsWindow::processIntern()
{
    if (ImGui::BeginTabBar("##Statistics", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::BeginTabItem("Timelines")) {
            if (ImGui::BeginChild("##timelines", ImVec2(0, 0), false)) {
                processTimelines();
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Histograms")) {
            if (ImGui::BeginChild("##histograms", ImVec2(0, 0), false)) {
                processHistograms();
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void _StatisticsWindow::processTimelines()
{
    ImGui::Spacing();
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Real time"), _live);
    ImGui::SameLine();
    ImGui::BeginDisabled(!_live);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scale(RightColumnWidth));
    ImGui::SliderFloat("", &_liveStatistics.history, 1, TimelineLiveStatistics::MaxLiveHistory, "%.1f s");
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (AlienImGui::Button("Export")) {
        onSaveStatistics();
    }
    AlienImGui::Separator();

    AlienImGui::Switcher(
        AlienImGui::SwitcherParameters()
            .name("Plot type")
            .textWidth(RightColumnWidth)
            .values(
            {"Accumulate values for all colors", "Break down by color", "Color #0", "Color #1", "Color #2", "Color #3", "Color #4", "Color #5", "Color #6"}),
        _plotType);
    ImGui::SameLine();
    if (ImGui::Button(_maximize ? ICON_FA_WINDOW_MINIMIZE : ICON_FA_WINDOW_MAXIMIZE)) {
        _maximize = !_maximize;
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    if (ImGui::BeginChild("##plots", ImVec2(0, 0), false)) {
        processTimelineStatistics();
    }
    ImGui::EndChild();
}

void _StatisticsWindow::processTimelineStatistics()
{
    ImGui::Spacing();
    AlienImGui::Group("Time step data");
    ImGui::PushID(1);
    int row = 0;
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_BordersInnerH, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, scale(RightColumnWidthTable));

        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numCells);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Cells");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numConnections);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Cell connections");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numParticles);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Energy particles");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::totalEnergy);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Total energy");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numSelfReplicators);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Self-replicators");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::averageGenomeCells, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Average genome size");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numViruses);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Viruses");

        ImPlot::PopColormap();

        ImGui::EndTable();
    }
    ImGui::PopID();

    ImGui::Spacing();
    AlienImGui::Group("Processes per time step and cell");
    ImGui::PushID(2);
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_BordersInnerH, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, scale(RightColumnWidthTable));
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numCreatedCells, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Created cells");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numAttacks, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Attacks");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numMuscleActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Muscle activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numTransmitterActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Transmitter activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numDefenderActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Defender activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numNervePulses, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Nerve pulses");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numNeuronActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Neuron activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numSensorActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Sensor activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numSensorMatches, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Sensor matches");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numInjectionActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Injection activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numCompletedInjections, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Completed injections");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numReconnectorCreated, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Reconnector creations");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numReconnectorRemoved, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Reconnector deletions");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numDetonations, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Detonations");

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
    ImGui::PopID();
}

void _StatisticsWindow::processHistograms()
{
    if (!_lastStatisticsData) {
        return;
    }
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha * 0.5 * Const::WindowAlpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha * 0.5 * Const::WindowAlpha));

    auto maxNumObjects = 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_HISTOGRAM_SLOTS; ++j) {
            auto value = _lastStatisticsData->histogram.numCellsByColorBySlot[i][j];
            maxNumObjects = std::max(maxNumObjects, value);
        }
    }

    //round maxNumObjects
    if (!_histogramUpperBound || toFloat(maxNumObjects) > *_histogramUpperBound * 0.9f || toFloat(maxNumObjects) < *_histogramUpperBound * 0.5f) {
        _histogramUpperBound = toFloat(maxNumObjects) * 1.3f;
    }

    ImPlot::SetNextPlotLimitsX(0, toFloat(MAX_HISTOGRAM_SLOTS), ImGuiCond_Always);
    ImPlot::SetNextPlotLimitsY(0, *_histogramUpperBound, ImGuiCond_Always);

    auto getLabelString = [](int value) {
        if (value >= 1000) {
            return std::to_string(value / 1000) + "K";
        } else {
            return std::to_string(value);
        }
    };

    //y-ticks
    char const* labelsY[6];
    double positionsY[6];
    for (int i = 0; i < 5; ++i) {
        labelsY[i] = "";
        positionsY[i] = *_histogramUpperBound / 5 * i;
    }
    auto temp = getLabelString(maxNumObjects);
    labelsY[5] = temp.c_str();
    positionsY[5] = toFloat(maxNumObjects);
    ImPlot::SetNextPlotTicksY(positionsY, 6, labelsY);

    //x-ticks
    char const* labelsX[5];
    std::string labelsX_temp[5];
    double positionsX[5];

    auto slotAge = _lastStatisticsData->histogram.maxValue / MAX_HISTOGRAM_SLOTS;
    for (int i = 0; i < 5; ++i) {
        labelsX_temp[i] = getLabelString(slotAge * ((MAX_HISTOGRAM_SLOTS - 1) / 4) * i);
        labelsX[i] = labelsX_temp[i].c_str();
        positionsX[i] = toFloat(((MAX_HISTOGRAM_SLOTS - 1) / 4) * i);
    }
    ImPlot::SetNextPlotTicksX(positionsX, 5, labelsX);
    ImPlot::SetNextPlotFormatX("");

    //plot histogram
    if (ImPlot::BeginPlot("##Histograms", "Age", "Cell count", ImVec2(-1, -1))) {

        auto const width = 1.0f / MAX_COLORS;
        for (int i = 0; i < MAX_COLORS; ++i) {
            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[i], h, s, v);
            ImPlot::PushStyleColor(ImPlotCol_Fill, (ImVec4)ImColor::HSV(h, s /** 3 / 4*/, v /** 3 / 4*/, ImGui::GetStyle().Alpha));
            ImPlot::PlotBars(
                (" ##" + std::to_string(i)).c_str(), _lastStatisticsData->histogram.numCellsByColorBySlot[i], MAX_HISTOGRAM_SLOTS, width, width * i);
            ImPlot::PopStyleColor(1);
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleColor(2);

}

void _StatisticsWindow::processPlot(int row, DataPoint DataPointCollection::*valuesPtr, int fracPartDecimals)
{
    auto count = _live ? toInt(_liveStatistics.dataPointCollectionHistory.size()) : toInt(_longtermStatistics.dataPointCollectionHistory.size());
    auto startTime = _live ? _liveStatistics.dataPointCollectionHistory.back().time - toDouble(_liveStatistics.history) : _longtermStatistics.dataPointCollectionHistory.front().time;
    auto endTime = _live ? _liveStatistics.dataPointCollectionHistory.back().time : _longtermStatistics.dataPointCollectionHistory.back().time;
    auto values = _live ? &(_liveStatistics.dataPointCollectionHistory[0].*valuesPtr) : &(_longtermStatistics.dataPointCollectionHistory[0].*valuesPtr);
    auto timePoints = _live ? &_liveStatistics.dataPointCollectionHistory[0].time : &_longtermStatistics.dataPointCollectionHistory[0].time;

    switch (_plotType) {
    case 0:
        plotSumColorsIntern(row, values, timePoints, count, startTime, endTime, fracPartDecimals);
        break;
    case 1:
        plotByColorIntern(row, values, timePoints, count, startTime, endTime, fracPartDecimals);
        break;
    default:
        plotForColorIntern(row, values, _plotType - 2, timePoints, count, startTime, endTime, fracPartDecimals);
        break;
    }
    ImGui::Spacing();
}

void _StatisticsWindow::processBackground()
{
    auto timestep = _simController->getCurrentTimestep();

    _lastStatisticsData = _simController->getStatistics();
    _liveStatistics.add(_lastStatisticsData->timeline, timestep);
    _longtermStatistics.add(_lastStatisticsData->timeline, timestep);
}

namespace
{
    double getMaxWithDataPointStride(double const* data, int count)
    {
        double result = 0;
        for (int i = count / 20; i < count; ++i) {
            result = std::max(result, *reinterpret_cast<double const*>(reinterpret_cast<DataPointCollection const*>(data) + i));
        }
        return result;
    }
}

void _StatisticsWindow::plotSumColorsIntern(
    int row,
    DataPoint const* dataPoint,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    double const* plotDataY = reinterpret_cast<double const*>(dataPoint) + MAX_COLORS;
    double upperBound = getMaxWithDataPointStride(plotDataY, count);
    double endValue = count > 0 ? *(reinterpret_cast<double const*>(reinterpret_cast<DataPointCollection const*>(dataPoint) + count - 1) + MAX_COLORS): 0.0;
    auto stride = toInt(sizeof(DataPointCollection));
    upperBound *= 1.5;
    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, scale(getPlotHeight())), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        auto color = ImPlot::GetColormapColor(row <= 10 ? row : 20 - row);
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                endTime, endValue, ImVec2(-10.0f, 10.0f), ImPlot::GetLastItemColor(), "%s", StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", timePoints, plotDataY, count, 0, stride);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", timePoints, plotDataY, count, 0, 0, stride);
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void _StatisticsWindow::plotByColorIntern(
    int row,
    DataPoint const* values,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto upperBound = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        upperBound = std::max(upperBound, getMaxWithDataPointStride(reinterpret_cast<double const*>(values) + i, count));
    }
    upperBound *= 1.5;

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);

    auto flags = _maximize ? ImPlotFlags_None : ImPlotFlags_NoLegend;
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, scale(getPlotHeight())), flags, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            ImGui::PushID(i);
            auto colorRaw = Const::IndividualCellColors[i];
            ImColor color(toInt((colorRaw >> 16) & 0xff), toInt((colorRaw >> 8) & 0xff), toInt(colorRaw & 0xff));

            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            auto endValue = count > 0 ? *(reinterpret_cast<double const*>(reinterpret_cast<DataPointCollection const*>(values) + (count - 1)) + i) : 0.0f;
            auto labelId = StringHelper::format(toFloat(endValue), fracPartDecimals);
            ImPlot::PlotLine(labelId.c_str(), timePoints, reinterpret_cast<double const*>(values) + i, count, 0, sizeof(DataPointCollection));
            ImPlot::PopStyleColor();
            ImGui::PopID();
        }

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar(2);
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void _StatisticsWindow::plotForColorIntern(
    int row,
    DataPoint const* values,
    int colorIndex,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto valuesForColor = reinterpret_cast<double const*>(values) + colorIndex;
    auto upperBound = getMaxWithDataPointStride(valuesForColor, count) * 1.5;
    auto endValue = count > 0 ? *reinterpret_cast<double const*>(reinterpret_cast<DataPointCollection const*>(valuesForColor) + count - 1) : 0.0;

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, scale(getPlotHeight())), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {

        float h, s, v;
        AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[colorIndex], h, s, v);
        auto color = static_cast<ImVec4>(ImColor::HSV(h, s, v));
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                endTime, endValue, ImVec2(-10.0f, 10.0f), ImPlot::GetLastItemColor(), "%s", StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", timePoints, valuesForColor, count, 0, sizeof(DataPointCollection));
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", timePoints, valuesForColor, count, 0, 0, sizeof(DataPointCollection));
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void _StatisticsWindow::onSaveStatistics()
{
    GenericFileDialogs::getInstance().showSaveFileDialog(
        "Export statistics", "Comma-separated values (*.csv){.csv},.*", _startingPath, [&](std::filesystem::path const& path) {
            auto firstFilename = ifd::FileDialog::Instance().GetResult();
            auto firstFilenameCopy = firstFilename;
            _startingPath = firstFilenameCopy.remove_filename().string();

            if (!ExportService::exportCollectedStatistics(_longtermStatistics.dataPointCollectionHistory, firstFilename.string())) {
                MessageDialog::getInstance().information("Export statistics", "The statistics could not be saved to the specified file.");
                return;
            }
        });
}

float _StatisticsWindow::getPlotHeight() const
{
    return _maximize ? PlotMaxHeight : PlotMinHeight;
}

