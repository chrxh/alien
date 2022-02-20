#include "StatisticsWindow.h"

#include <imgui.h>
#include <implot.h>

#include "Base/StringHelper.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/OverallStatistics.h"
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
    template<typename T>
    T getMax(std::vector<T> const& range)
    {
        T result = static_cast<T>(0);
        for (auto const& element : range) {
            if (element > result) {
                result = element;
            }
        }
        return result;
    }
}

void _StatisticsWindow::reset()
{
    _liveStatistics = LiveStatistics();
    _longtermStatistics = LongtermStatistics();
}

void _StatisticsWindow::processIntern()
{
    _exportStatisticsDialog->process();

    AlienImGui::ToggleButton("Real time", _live);

    ImGui::SameLine();
    ImGui::BeginDisabled(!_live);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - StyleRepository::getInstance().scaleContent(60));
    ImGui::SliderFloat("", &_liveStatistics.history, 1, LiveStatistics::MaxLiveHistory, "%.1f s");
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
    if (ImGui::BeginTable(
            "##",
            2,
            /*ImGuiTableFlags_BordersV | */ ImGuiTableFlags_RowBg
                | ImGuiTableFlags_BordersOuter,
            ImVec2(- 1, 0))) {
        ImGui::TableSetupColumn(
            "Entities", ImGuiTableColumnFlags_WidthFixed, StyleRepository::getInstance().scaleContent(125.0f));
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Cells");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(0, _liveStatistics.datas[0]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Energy particles");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(1, _liveStatistics.datas[1]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Tokens");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(2, _liveStatistics.datas[2]);
        ImPlot::PopColormap();
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::BeginTable(
            "##",
            2,
            /*ImGuiTableFlags_BordersV | */ ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter,
            ImVec2(-1, 0))) {
        ImGui::TableSetupColumn(
            "Processes", ImGuiTableColumnFlags_WidthFixed, StyleRepository::getInstance().scaleContent(125.0f));
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Created cells");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(3, _liveStatistics.datas[3]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Successful attacks");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(4, _liveStatistics.datas[4]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Failed attacks");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(5, _liveStatistics.datas[5]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text("Muscle activities");
        ImGui::TableSetColumnIndex(1);
        processLivePlot(6, _liveStatistics.datas[6]);

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
        ImGui::TableSetupColumn("Entities", ImGuiTableColumnFlags_WidthFixed, 125.0f);
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Cells");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(0, _longtermStatistics.datas[0]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Energy particles");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(1, _longtermStatistics.datas[1]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Tokens");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(2, _longtermStatistics.datas[2]);
        ImPlot::PopColormap();
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::BeginTable(
            "##",
            2,
            /*ImGuiTableFlags_BordersV | */ ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter,
            ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("Processes", ImGuiTableColumnFlags_WidthFixed, 125.0f);
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Created cells");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(3, _longtermStatistics.datas[3]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Successful attacks");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(4, _longtermStatistics.datas[4]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Failed attacks");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(5, _longtermStatistics.datas[5]);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Muscle activities");
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(6, _longtermStatistics.datas[6]);

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
}

void _StatisticsWindow::processLivePlot(int row, std::vector<float> const& valueHistory)
{
    auto maxValue = getMax(valueHistory);
    
    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(
        _liveStatistics.timepointsHistory.back() - _liveStatistics.history,
        _liveStatistics.timepointsHistory.back(),
        0,
        maxValue * 1.5,
        ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, 80), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        auto color = ImPlot::GetColormapColor(row + 2);

        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                _liveStatistics.timepointsHistory.back(),
                valueHistory.back(),
                ImVec2(-10.0f, 10.0f),
                color,
                "%s",
                StringHelper::format(toInt(valueHistory.back())).c_str());
        }

        ImPlot::PushStyleColor(ImPlotCol_Line, color);
        ImPlot::PlotLine(
            "##", _liveStatistics.timepointsHistory.data(), valueHistory.data(), toInt(valueHistory.size()));

        if (row > 0) {
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", _liveStatistics.timepointsHistory.data(), valueHistory.data(), toInt(valueHistory.size()));
            ImPlot::PopStyleVar();
        }
        ImPlot::PopStyleColor();

        if (row == 0) {
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.9;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor1 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.8;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor2 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.7;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor3 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.6;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor4 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.5;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor5 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.4;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor6 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
            {
                std::vector<float> temp = valueHistory;
                for (auto& t : temp) {
                    t *= 0.3;
                }
                ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)ImColor(Const::IndividualCellColor7 | 0xff000000));
                ImPlot::PlotLine("##", _liveStatistics.timepointsHistory.data(), temp.data(), toInt(valueHistory.size()));
                ImPlot::PopStyleColor();
            }
        }

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void _StatisticsWindow::processLongtermPlot(int row, std::vector<float> const& valueHistory)
{
    auto maxValue = getMax(valueHistory);

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(
        _longtermStatistics.timestepHistory.front(),
        _longtermStatistics.timestepHistory.back(),
        0,
        maxValue * 1.5,
        ImGuiCond_Always);  
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, 80), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        auto color = ImPlot::GetColormapColor(row + 2);
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::AnnotateClamped(
                _longtermStatistics.timestepHistory.back(),
                valueHistory.back(),
                ImVec2(-10.0f, 10.0f),
                ImPlot::GetLastItemColor(),
                "%s",
                StringHelper::format(toInt(valueHistory.back())).c_str());
        }
        ImPlot::PushStyleColor(ImPlotCol_Line, color);
        ImPlot::PlotLine(
            "##", _longtermStatistics.timestepHistory.data(), valueHistory.data(), toInt(valueHistory.size()));
        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::PlotShaded(
            "##", _longtermStatistics.timestepHistory.data(), valueHistory.data(), toInt(valueHistory.size()));
        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void _StatisticsWindow::processBackground()
{
    auto newStatistics = _simController->getStatistics();
    _liveStatistics.add(newStatistics);

    _longtermStatistics.add(newStatistics);
}
