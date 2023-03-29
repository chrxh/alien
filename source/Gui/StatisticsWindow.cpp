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

    template<typename T>
    T getMax(T const* data, int count)
    {
        T result = static_cast<T>(0);
        for (int i = 0; i < count; ++i) {
            result = std::max(result, *reinterpret_cast<T const*>(reinterpret_cast<DataPoint const*>(data) + i));
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
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - StyleRepository::getInstance().contentScale(60));
    ImGui::SliderFloat("", &_liveStatistics.history, 1, TimelineLiveStatistics::MaxLiveHistory, "%.1f s");
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (AlienImGui::Button("Export")) {
        _exportStatisticsDialog->show(_longtermStatistics);
    }

    if (_live) {
        processLiveStatistics();
    } else {
        processLongtermStatistics();
    }
}

void _StatisticsWindow::processLiveStatistics()
{
    ImGui::Spacing();
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("Objects", ImGuiTableColumnFlags_WidthFixed, StyleRepository::getInstance().contentScale(HeadColWidth));
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();

        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Cells");
        auto text = _showCellsByColor ? ICON_FA_MINUS_SQUARE : ICON_FA_PLUS_SQUARE;
        if (AlienImGui::Button(text)) {
            _showCellsByColor = !_showCellsByColor;
        }

        ImGui::TableSetColumnIndex(1);
        processLivePlot(0, &_liveStatistics.dataPoints[0].numCells);
        if (_showCellsByColor) {
            processLivePlotForCellsByColor(1, &_liveStatistics.dataPoints[0].numCellsByColor);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Cell connections");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(2, &_liveStatistics.dataPoints[0].numConnections);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Energy particles");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(3, &_liveStatistics.dataPoints[0].numParticles);

        ImPlot::PopColormap();

        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("Processes per time step", ImGuiTableColumnFlags_WidthFixed, StyleRepository::getInstance().contentScale(HeadColWidth));
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Created cells");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(4, &_liveStatistics.dataPoints[0].numCreatedCells, 2);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Successful attacks");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(5, &_liveStatistics.dataPoints[0].numSuccessfulAttacks, 2);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Failed attacks");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(6, &_liveStatistics.dataPoints[0].numFailedAttacks, 2);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Muscle activities");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(7, &_liveStatistics.dataPoints[0].numMuscleActivities, 2);

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

void _StatisticsWindow::processLongtermStatistics()
{
    ImGui::Spacing();
    if (ImGui::BeginTable(
            "##",
            2,
            /*ImGuiTableFlags_BordersV | */ ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter,
            ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("Objects", ImGuiTableColumnFlags_WidthFixed, StyleRepository::getInstance().contentScale(HeadColWidth));
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Cells");
        auto text = _showCellsByColor ? ICON_FA_MINUS_SQUARE : ICON_FA_PLUS_SQUARE;
        if (AlienImGui::Button(text)) {
            _showCellsByColor = !_showCellsByColor;
        }

        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(0, &_longtermStatistics.dataPoints[0].numCells);
        if (_showCellsByColor) {
            processLongtermPlotForCellsByColor(1, &_longtermStatistics.dataPoints[0].numCellsByColor);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Cell connections");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(2, &_longtermStatistics.dataPoints[0].numConnections);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Energy particles");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(3, &_longtermStatistics.dataPoints[0].numParticles);

        ImPlot::PopColormap();
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::BeginTable(
            "##",
            2,
            /*ImGuiTableFlags_BordersV | */ ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter,
            ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("Processes per time step", ImGuiTableColumnFlags_WidthFixed, StyleRepository::getInstance().contentScale(HeadColWidth));
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Created cells");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(4, &_longtermStatistics.dataPoints[0].numCreatedCells, 2);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Successful attacks");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(5, &_longtermStatistics.dataPoints[0].numSuccessfulAttacks, 2);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Failed attacks");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(6, &_longtermStatistics.dataPoints[0].numFailedAttacks, 2);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Muscle activities");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(7, &_longtermStatistics.dataPoints[0].numMuscleActivities, 2);

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
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
    if (ImPlot::BeginPlot("##Histograms", "Age", "Number of objects", ImVec2(-1, -1))) {

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

void _StatisticsWindow::processLivePlot(int colorIndex, double const* data, int fracPartDecimals)
{
    auto count = toInt(_liveStatistics.dataPoints.size());
    auto endTime = _liveStatistics.dataPoints.back().time;
    plotIntern(colorIndex, data, &_liveStatistics.dataPoints[0].time, count, endTime - toDouble(_liveStatistics.history), endTime, fracPartDecimals);
}

void _StatisticsWindow::processLivePlotForCellsByColor(int colorIndex, ColorVector<double> const* data)
{
    auto count = toInt(_liveStatistics.dataPoints.size());
    auto startTime = _liveStatistics.dataPoints.back().time - toDouble(_liveStatistics.history);
    auto endTime = _liveStatistics.dataPoints.back().time;
    plotByColorIntern(colorIndex, data, &_liveStatistics.dataPoints[0].time, count, startTime, endTime);
}


void _StatisticsWindow::processLongtermPlot(int colorIndex, double const* data, int fracPartDecimals)
{
    auto count = toInt(_longtermStatistics.dataPoints.size());
    auto startTime = _longtermStatistics.dataPoints.front().time;
    auto endTime = _longtermStatistics.dataPoints.back().time;
    plotIntern(colorIndex, data, &_longtermStatistics.dataPoints[0].time, count, startTime, endTime, fracPartDecimals);
}

void _StatisticsWindow::processLongtermPlotForCellsByColor(int colorIndex, ColorVector<double> const* data)
{
    auto count = toInt(_longtermStatistics.dataPoints.size());
    auto startTime = _longtermStatistics.dataPoints.front().time;
    auto endTime = _longtermStatistics.dataPoints.back().time;
    plotByColorIntern(colorIndex, data, &_longtermStatistics.dataPoints[0].time, count, startTime, endTime);
}

void _StatisticsWindow::processBackground()
{
    auto timestep = _simController->getCurrentTimestep();

    _lastStatisticsData = _simController->getStatistics();
    _liveStatistics.add(_lastStatisticsData->timeline, timestep);
    _longtermStatistics.add(_lastStatisticsData->timeline, timestep);
}

void _StatisticsWindow::plotIntern(
    int colorIndex,
    double const* data,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto upperBound = getMax(data, count) * 1.5;
    auto endValue = count > 0 ? *reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(data) + (count - 1)) : 0.0;

    ImGui::PushID(colorIndex);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, contentScale(80.0f)), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        auto color = ImPlot::GetColormapColor(colorIndex + 2);
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                endTime, endValue, ImVec2(-10.0f, 10.0f), ImPlot::GetLastItemColor(), "%s", StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", timePoints, data, count, 0, sizeof(DataPoint));
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", timePoints, data, count, 0, 0, sizeof(DataPoint));
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void _StatisticsWindow::plotByColorIntern(int colorIndex, ColorVector<double> const* data, double const* timePoints, int count, double startTime, double endTime)
{
    auto upperBound = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        upperBound = std::max(upperBound, getMax(reinterpret_cast<double const*>(data) + i, count));
    }
    upperBound *= 1.5;

    ImGui::PushID(colorIndex);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
    ImPlot::SetNextPlotLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, contentScale(160.0f)), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            ImGui::PushID(i);
            auto colorRaw = Const::IndividualCellColors[i];
            ImColor color(toInt((colorRaw >> 16) & 0xff), toInt((colorRaw >> 8) & 0xff), toInt(colorRaw & 0xff));

            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            auto endValue = count > 0 ? *(reinterpret_cast<double const*>(reinterpret_cast<DataPoint const*>(data) + (count - 1)) + i) : 0.0f;
            auto labelId = std::to_string(toInt(endValue));
            ImPlot::PlotLine(labelId.c_str(), timePoints, reinterpret_cast<double const*>(data) + i, count, 0, sizeof(DataPoint));
            ImPlot::PopStyleColor();
            ImGui::PopID();
        }

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar(2);
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

