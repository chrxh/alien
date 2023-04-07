#include "StatisticsWindow.h"

#include <imgui.h>
#include <implot.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationController.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"
#include "ExportStatisticsDialog.h"

_StatisticsWindow::_StatisticsWindow(SimulationController const& simController)
    : _AlienWindow("Statistics", "windows.statistics", false)
    , _simController(simController)
{
    _exportStatisticsDialog = std::make_shared<_ExportStatisticsDialog>();
}

namespace
{
    auto const HeadColWidth = 150.0f;

    double getMax(double const* data, int count)
    {
        double result = 0;
        for (int i = 0; i < count; ++i) {
            result = std::max(result, *reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(data) + i));
        }
        return result;
    }
}

void _StatisticsWindow::reset()
{
    _liveStatistics = TimelineLiveStatistics();
    _longtermStatistics = TimelineLongtermStatistics();
}

void _StatisticsWindow::processIntern()
{
    _exportStatisticsDialog->process();

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
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - contentScale(HeadColWidth));
    ImGui::SliderFloat("", &_liveStatistics.history, 1, TimelineLiveStatistics::MaxLiveHistory, "%.1f s");
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (AlienImGui::Button("Export")) {
        _exportStatisticsDialog->show(_longtermStatistics);
    }
    AlienImGui::Separator();

    AlienImGui::Switcher(
        AlienImGui::SwitcherParameters()
            .name("Plot type")
            .textWidth(HeadColWidth)
            .values(
            {"Accumulate values for all colors", "Break down by color", "Color #0", "Color #1", "Color #2", "Color #3", "Color #4", "Color #5", "Color #6"}),
        _plotType);

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
    AlienImGui::Group("Objects count");
    ImGui::PushID(1);
    if (ImGui::BeginTable("##", 2, 0, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, contentScale(HeadColWidth));

        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(0, &DataPoint::numCells);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Cells");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(1, &DataPoint::numConnections);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Cell connections");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(2, &DataPoint::numParticles);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Energy particles");

        ImPlot::PopColormap();

        ImGui::EndTable();
    }
    ImGui::PopID();

    ImGui::Spacing();
    AlienImGui::Group("Processes per time step");
    ImGui::PushID(2);
    if (ImGui::BeginTable("##", 2, 0, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, contentScale(HeadColWidth));
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(3, &DataPoint::numCreatedCells, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Created cells");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(4, &DataPoint::numAttacks, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Attacks");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(5, &DataPoint::numMuscleActivities, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Muscle activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(6, &DataPoint::numTransmitterActivities, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Transmitter activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(7, &DataPoint::numDefenderActivities, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Defender activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(8, &DataPoint::numInjectionActivities, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Injection activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(9, &DataPoint::numCompletedInjections, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Completed injections");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(10, &DataPoint::numNervePulses, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Nerve pulses");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(11, &DataPoint::numNeuronActivities, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Neuron activities");

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
    if (ImPlot::BeginPlot("##Histograms", "Age", "Cells count", ImVec2(-1, -1))) {

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

void _StatisticsWindow::processPlot(int row, ColorVector<double> DataPoint::*valuesPtr, int fracPartDecimals)
{
    auto count = _live ? toInt(_liveStatistics.dataPoints.size()) : toInt(_longtermStatistics.dataPoints.size());
    auto startTime = _live ? _liveStatistics.dataPoints.back().time - toDouble(_liveStatistics.history) : _longtermStatistics.dataPoints.front().time;
    auto endTime = _live ? _liveStatistics.dataPoints.back().time : _longtermStatistics.dataPoints.back().time;
    auto values = _live ? &(_liveStatistics.dataPoints[0].*valuesPtr) : &(_longtermStatistics.dataPoints[0].*valuesPtr);
    auto timePoints = _live ? &_liveStatistics.dataPoints[0].time : &_longtermStatistics.dataPoints[0].time;

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
}

void _StatisticsWindow::processBackground()
{
    auto timestep = _simController->getCurrentTimestep();

    _lastStatisticsData = _simController->getStatistics();
    _liveStatistics.add(_lastStatisticsData->timeline, timestep);
    _longtermStatistics.add(_lastStatisticsData->timeline, timestep);
}

void _StatisticsWindow::plotSumColorsIntern(
    int row,
    ColorVector<double> const* values,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto& accumulatedData = _cachedTimelines[0];
    auto& rearragedtimePoints = _cachedTimelines[1];
    accumulatedData.resize(count);
    rearragedtimePoints.resize(count);
    double upperBound = 0;
    for (int i = 0; i < count; ++i) {
        double sum = 0;
        auto ptr = reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(values) + i); 
        for (int color = 0; color < MAX_COLORS; ++color) {
            sum += *(ptr + color);
        }
        accumulatedData[i] = sum;
        upperBound = std::max(upperBound, sum);
        rearragedtimePoints[i] = *reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(timePoints) + i); 
    }
    auto endValue = count > 0 ? accumulatedData.back() : 0.0;
    upperBound *= 1.5f;
    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, contentScale(80.0f)), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        auto color = ImPlot::GetColormapColor(std::min(10, row));
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                endTime, endValue, ImVec2(-10.0f, 10.0f), ImPlot::GetLastItemColor(), "%s", StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", rearragedtimePoints.data(), accumulatedData.data(), count);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", rearragedtimePoints.data(), accumulatedData.data(), count);
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
    ColorVector<double> const* values,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto upperBound = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        upperBound = std::max(upperBound, getMax(reinterpret_cast<double const*>(values) + i, count));
    }
    upperBound *= 1.5;

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, contentScale(160.0f)), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            ImGui::PushID(i);
            auto colorRaw = Const::IndividualCellColors[i];
            ImColor color(toInt((colorRaw >> 16) & 0xff), toInt((colorRaw >> 8) & 0xff), toInt(colorRaw & 0xff));

            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            auto endValue = count > 0 ? *(reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(values) + (count - 1)) + i) : 0.0f;
            auto labelId = StringHelper::format(toFloat(endValue), fracPartDecimals);
            ImPlot::PlotLine(labelId.c_str(), timePoints, reinterpret_cast<double const*>(values) + i, count, 0, sizeof(DataPoint));
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
    ColorVector<double> const* values,
    int colorIndex,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto valuesForColor = reinterpret_cast<double const*>(values) + colorIndex;
    auto upperBound = getMax(valuesForColor, count) * 1.5;
    auto endValue = count > 0 ? *reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(valuesForColor) + count - 1) : 0.0;

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, contentScale(80.0f)), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {

        float h, s, v;
        AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[colorIndex], h, s, v);
        auto color = static_cast<ImVec4>(ImColor::HSV(h, s, v));
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                endTime, endValue, ImVec2(-10.0f, 10.0f), ImPlot::GetLastItemColor(), "%s", StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", timePoints, valuesForColor, count, 0, sizeof(DataPoint));
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", timePoints, valuesForColor, count, 0, 0, sizeof(DataPoint));
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

