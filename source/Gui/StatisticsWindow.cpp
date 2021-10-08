#include "StatisticsWindow.h"

#include "imgui.h"
#include "implot.h"

#include "EngineInterface/OverallStatistics.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"

namespace
{
    float const MaxLiveHistory = 120.0f; //in seconds
    float const LongtermTimestepDelta = 1000.0f;
}

_StatisticsWindow::_StatisticsWindow(SimulationController const& simController)
    : _simController(simController)
{
    ImPlot::GetStyle().AntiAliasedLines = true;
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
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Statistics", &_on, windowFlags);

    ImGui::Checkbox("Real-time", &_live);

    ImGui::SameLine();
    ImGui::BeginDisabled(!_live);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderFloat("", &_liveStatistics.history, 1, MaxLiveHistory, "%.1f s");
    ImGui::EndDisabled();

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
    auto maxCells = getMax(_liveStatistics.numCellsHistory);
    auto maxParticles = getMax(_liveStatistics.numParticlesHistory);
    auto maxTokens = getMax(_liveStatistics.numTokensHistory);

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
        ImPlot::PushColormap(ImPlotColormap_Plasma);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Cells", 0);
        ImGui::TableSetColumnIndex(1);
        processLivePlot(0, _liveStatistics.numCellsHistory);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Particles", 0);
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
        ImPlot::PushColormap(ImPlotColormap_Plasma);

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
    auto maxCells = getMax(_longtermStatistics.numCellsHistory);
    auto maxParticles = getMax(_longtermStatistics.numParticlesHistory);
    auto maxTokens = getMax(_longtermStatistics.numTokensHistory);

    static float rratios[] = {1, 1};
    static float cratios[] = {1, 1};
    if (ImPlot::BeginSubplots("##", 2, 1, ImVec2(-1, -1), ImPlotSubplotFlags_LinkCols, rratios, cratios)) {
        ImPlot::FitNextPlotAxes(true, true);
/*
        ImPlot::SetNextPlotLimits(
            _longtermStatistics.timestepHistory.front(),
            _longtermStatistics.timestepHistory.back(),
            0,
            std::max(maxCells, maxParticles) * 1.5,
            ImGuiCond_Appearing);
*/
        static ImPlotAxisFlags flags = 0;
        if (ImPlot::BeginPlot(
                "##Entities",
                "time step",
                NULL,
                ImGui::GetContentRegionAvail(),
                //            ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2),
                0,
                flags,
                flags)) {
            float labelPosY =
                _longtermStatistics.numCellsHistory.back() > _longtermStatistics.numParticlesHistory.back() ? -10.0f
                                                                                                            : 10.0f;

            ImPlot::PlotLine(
                "Cells",
                _longtermStatistics.timestepHistory.data(),
                _longtermStatistics.numCellsHistory.data(),
                toInt(_longtermStatistics.numCellsHistory.size()));
            ImPlot::AnnotateClamped(
                _longtermStatistics.timestepHistory.back(),
                _longtermStatistics.numCellsHistory.back(),
                ImVec2(-10.0f, labelPosY),
                ImPlot::GetLastItemColor(),
                std::to_string(toInt(_longtermStatistics.numCellsHistory.back())).c_str());

            ImPlot::PlotLine(
                "Energy particles",
                _longtermStatistics.timestepHistory.data(),
                _longtermStatistics.numParticlesHistory.data(),
                toInt(_longtermStatistics.numParticlesHistory.size()));
            ImPlot::AnnotateClamped(
                _longtermStatistics.timestepHistory.back(),
                _longtermStatistics.numParticlesHistory.back(),
                ImVec2(-10.0f, -labelPosY),
                ImPlot::GetLastItemColor(),
                std::to_string(toInt(_longtermStatistics.numParticlesHistory.back())).c_str());

            ImPlot::PlotLine(
                "Tokens",
                _longtermStatistics.timestepHistory.data(),
                _longtermStatistics.numTokensHistory.data(),
                toInt(_longtermStatistics.numTokensHistory.size()));
            ImPlot::AnnotateClamped(
                _longtermStatistics.timestepHistory.back(),
                _longtermStatistics.numTokensHistory.back(),
                ImVec2(-10, 10),
                ImPlot::GetLastItemColor(),
                std::to_string(toInt(_longtermStatistics.numTokensHistory.back())).c_str());

            ImPlot::EndPlot();
        }

//        ImPlot::FitNextPlotAxes(true, true);
/*
        ImPlot::SetNextPlotLimits(
            _longtermStatistics.timestepHistory.front(),
            _longtermStatistics.timestepHistory.back(),
            0,
            maxTokens * 1.5,
            ImGuiCond_Appearing);
*/
        if (ImPlot::BeginPlot(
                "##Tokens",
                "time step",
                NULL,
                ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y),
                0,
                flags,
                flags)) {

            ImPlot::PlotLine(
                "Tokens",
                _longtermStatistics.timestepHistory.data(),
                _longtermStatistics.numTokensHistory.data(),
                toInt(_longtermStatistics.numTokensHistory.size()));
            ImPlot::AnnotateClamped(
                _longtermStatistics.timestepHistory.back(),
                _longtermStatistics.numTokensHistory.back(),
                ImVec2(-10, 10),
                ImPlot::GetLastItemColor(),
                std::to_string(toInt(_longtermStatistics.numTokensHistory.back())).c_str());

            ImPlot::PlotLine(
                "Created cells",
                _longtermStatistics.timestepHistory.data(),
                _longtermStatistics.numCreatedCellsHistory.data(),
                toInt(_longtermStatistics.numCreatedCellsHistory.size()));
            ImPlot::AnnotateClamped(
                _longtermStatistics.timestepHistory.back(),
                _longtermStatistics.numCreatedCellsHistory.back(),
                ImVec2(-10, 10),
                ImPlot::GetLastItemColor(),
                std::to_string(toInt(_longtermStatistics.numCreatedCellsHistory.back())).c_str());

            ImPlot::EndPlot();
        }

        ImPlot::EndSubplots();
    }
}

void _StatisticsWindow::processLivePlot(int row, std::vector<float> const& valueHistory)
{
    auto maxValue = getMax(valueHistory);
    
    ImGui::PushID(row);
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextPlotLimits(
        _liveStatistics.timepointsHistory.back() - _liveStatistics.history,
        _liveStatistics.timepointsHistory.back(),
        0,
        maxValue * 1.5,
        ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", 0, 0, ImVec2(-1, 80), 0, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels)) {
        ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(row));

        ImPlot::PlotLine(
            "##",
            _liveStatistics.timepointsHistory.data(),
            valueHistory.data(),
            toInt(valueHistory.size()));

        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::PlotShaded(
            "##",
            _liveStatistics.timepointsHistory.data(),
            valueHistory.data(),
            toInt(valueHistory.size()));
        ImPlot::AnnotateClamped(
            _liveStatistics.timepointsHistory.back(),
            valueHistory.back(),
            ImVec2(-10.0f, 10.0f),
            ImPlot::GetLastItemColor(),
            std::to_string(toInt(valueHistory.back())).c_str());

        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
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
        numSuccessfullAttacksHistory.emplace_back(toFloat(newStatistics.numSuccessfulAttacks));
        numFailedAttacksHistory.emplace_back(toFloat(newStatistics.numFailedAttacks));
        numMuscleActivitiesHistory.emplace_back(toFloat(newStatistics.numMuscleActivities));
    }
}
