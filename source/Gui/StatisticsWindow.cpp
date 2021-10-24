#include "StatisticsWindow.h"

#include "imgui.h"
#include "implot.h"

#include "Base/StringFormatter.h"
#include "EngineInterface/OverallStatistics.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"

namespace
{
    float const MaxLiveHistory = 120.0f; //in seconds
    float const LongtermTimestepDelta = 1000.0f;
}

_StatisticsWindow::_StatisticsWindow(SimulationController const& simController)
    : _simController(simController)
{
    ImPlot::GetStyle().AntiAliasedLines = true;
    _on = GlobalSettings::getInstance().getBoolState("windows.statistics.active", true);
}

_StatisticsWindow::~_StatisticsWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.statistics.active", _on);
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

void _StatisticsWindow::process()
{
    updateData();

    if (!_on) {
        return;
    }

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::Begin("Statistics", &_on, windowFlags);

    ImGui::Checkbox("Real time", &_live);

    ImGui::SameLine();
    ImGui::BeginDisabled(!_live);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 60);
    ImGui::SliderFloat("", &_liveStatistics.history, 1, MaxLiveHistory, "%.1f s");
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Button("Export");

    if (_live) {
        processLiveStatistics();
    } else {
        processLongtermStatistics();
    }

    ImGui::End();
//    ImPlot::ShowDemoWindow();
}

bool _StatisticsWindow::isOn() const
{
    return _on;
}

void _StatisticsWindow::setOn(bool value)
{
    _on = value;
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
        ImGui::TableSetupColumn("Entities", ImGuiTableColumnFlags_WidthFixed, 125.0f);
        ImGui::TableSetupColumn("##");
        ImGui::TableHeadersRow();
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Cells", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(0, _liveStatistics.numCellsHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Energy particles", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(1, _liveStatistics.numParticlesHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Tokens", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(2, _liveStatistics.numTokensHistory);
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
        ImGui::Text("Created cells", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(3, _liveStatistics.numCreatedCellsHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Successful attacks", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(4, _liveStatistics.numSuccessfulAttacksHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Failed attacks", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(5, _liveStatistics.numFailedAttacksHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Muscle activities", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(6, _liveStatistics.numMuscleActivitiesHistory);

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
        ImGui::Text("Cells", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(0, _longtermStatistics.numCellsHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Energy particles", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(1, _longtermStatistics.numParticlesHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Tokens", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(2, _longtermStatistics.numTokensHistory);
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
        ImGui::Text("Created cells", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(3, _longtermStatistics.numCreatedCellsHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Successful attacks", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(4, _longtermStatistics.numSuccessfulAttacksHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Failed attacks", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(5, _longtermStatistics.numFailedAttacksHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Muscle activities", 0);
        ImGui::TableSetColumnIndex(1);
        processLongtermPlot(6, _longtermStatistics.numMuscleActivitiesHistory);

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
                StringFormatter::format(toInt(valueHistory.back())).c_str());
        }

        ImPlot::PushStyleColor(ImPlotCol_Line, color);
        ImPlot::PlotLine(
            "##", _liveStatistics.timepointsHistory.data(), valueHistory.data(), toInt(valueHistory.size()));

        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f * ImGui::GetStyle().Alpha);
        ImPlot::PlotShaded(
            "##", _liveStatistics.timepointsHistory.data(), valueHistory.data(), toInt(valueHistory.size()));
        ImPlot::PopStyleVar();

        ImPlot::PopStyleColor();

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
                StringFormatter::format(toInt(valueHistory.back())).c_str());
        }
        ImPlot::PushStyleColor(ImPlotCol_Line, color);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle);
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

void _StatisticsWindow::updateData()
{
    auto newStatistics = _simController->getStatistics();
    _liveStatistics.add(newStatistics);

    _longtermStatistics.add(newStatistics);
}

void _StatisticsWindow::LiveStatistics::truncate()
{
    if (!timepointsHistory.empty()
        && timepointsHistory.back() - timepointsHistory.front() > (MaxLiveHistory + 1.0f)) {
        timepointsHistory.erase(timepointsHistory.begin());
        numCellsHistory.erase(numCellsHistory.begin());
        numParticlesHistory.erase(numParticlesHistory.begin());
        numTokensHistory.erase(numTokensHistory.begin());
        numCreatedCellsHistory.erase(numCreatedCellsHistory.begin());
        numSuccessfulAttacksHistory.erase(numSuccessfulAttacksHistory.begin());
        numFailedAttacksHistory.erase(numFailedAttacksHistory.begin());
        numMuscleActivitiesHistory.erase(numMuscleActivitiesHistory.begin());
    }
}

void _StatisticsWindow::LiveStatistics::add(OverallStatistics const& newStatistics)
{
    truncate();

    timepoint += ImGui::GetIO().DeltaTime;
    timepointsHistory.emplace_back(timepoint);
    numCellsHistory.emplace_back(toFloat(newStatistics.numCells));
    numParticlesHistory.emplace_back(toFloat(newStatistics.numParticles));
    numTokensHistory.emplace_back(toFloat(newStatistics.numTokens));
    numCreatedCellsHistory.emplace_back(toFloat(newStatistics.numCreatedCells));
    numSuccessfulAttacksHistory.emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
    numFailedAttacksHistory.emplace_back(toFloat(newStatistics.numFailedAttacks));
    numMuscleActivitiesHistory.emplace_back(toFloat(newStatistics.numMuscleActivities));
}

void _StatisticsWindow::LongtermStatistics::add(OverallStatistics const& newStatistics)
{
    if (timestepHistory.empty()
        || newStatistics.timeStep - timestepHistory.back() > LongtermTimestepDelta) {
        timestepHistory.emplace_back(toFloat(newStatistics.timeStep));
        numCellsHistory.emplace_back(toFloat(newStatistics.numCells));
        numParticlesHistory.emplace_back(toFloat(newStatistics.numParticles));
        numTokensHistory.emplace_back(toFloat(newStatistics.numTokens));
        numCreatedCellsHistory.emplace_back(toFloat(newStatistics.numCreatedCells));
        numSuccessfulAttacksHistory.emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
        numFailedAttacksHistory.emplace_back(toFloat(newStatistics.numFailedAttacks));
        numMuscleActivitiesHistory.emplace_back(toFloat(newStatistics.numMuscleActivities));
    }
}
