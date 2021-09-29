#include "StatisticsWindow.h"

#include "imgui.h"
#include "implot.h"

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

    ImGui::Checkbox("Live", &_live);

    ImGui::SameLine();
    ImGui::BeginDisabled(!_live);
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

    ImPlot::SetNextPlotLimits(
        _liveStatistics.timepointsHistory.back() - _liveStatistics.history,
        _liveStatistics.timepointsHistory.back(),
        0,
        std::max(maxCells, maxParticles) * 1.5,
        ImGuiCond_Always);
    static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickLabels;
    ImPlot::BeginPlot(
        "",
        NULL,
        NULL,
        ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y * 0.5f),
        0,
        flags,
        flags);
    float labelPosY =
        _liveStatistics.numCellsHistory.back() > _liveStatistics.numParticlesHistory.back() ? -10.0f : 10.0f;

    ImPlot::PlotLine("Cells", _liveStatistics.timepointsHistory.data(), _liveStatistics.numCellsHistory.data(), toInt(_liveStatistics.numCellsHistory.size()));
    ImPlot::AnnotateClamped(
        _liveStatistics.timepointsHistory.back(),
        _liveStatistics.numCellsHistory.back(),
        ImVec2(-10.0f, labelPosY),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_liveStatistics.numCellsHistory.back())).c_str());

    ImPlot::PlotLine(
        "Energy particles", _liveStatistics.timepointsHistory.data(), _liveStatistics.numParticlesHistory.data(), toInt(_liveStatistics.numParticlesHistory.size()));
    ImPlot::AnnotateClamped(
        _liveStatistics.timepointsHistory.back(),
        _liveStatistics.numParticlesHistory.back(),
        ImVec2(-10.0f, -labelPosY),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_liveStatistics.numParticlesHistory.back())).c_str());

    ImPlot::EndPlot();

    ImPlot::SetNextPlotLimits(
        _liveStatistics.timepointsHistory.back() - _liveStatistics.history, _liveStatistics.timepointsHistory.back(), 0, maxTokens * 1.5, ImGuiCond_Always);
    ImPlot::BeginPlot(
        "##Tokens",
        NULL,
        NULL,
        ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y),
        0,
        flags,
        flags);

    ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(2));
    ImPlot::PlotLine("Tokens", _liveStatistics.timepointsHistory.data(), _liveStatistics.numTokensHistory.data(), toInt(_liveStatistics.numTokensHistory.size()));
    ImPlot::AnnotateClamped(
        _liveStatistics.timepointsHistory.back(),
        _liveStatistics.numTokensHistory.back(),
        ImVec2(-10, 10),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_liveStatistics.numTokensHistory.back())).c_str());
    ImPlot::PopStyleColor();

    ImPlot::EndPlot();
}

void _StatisticsWindow::processLongtermStatistics()
{
    auto maxCells = getMax(_longtermStatistics.numCellsHistory);
    auto maxParticles = getMax(_longtermStatistics.numParticlesHistory);
    auto maxTokens = getMax(_longtermStatistics.numTokensHistory);

    ImPlot::SetNextPlotLimits(
        _longtermStatistics.timestepHistory.front(),
        _longtermStatistics.timestepHistory.back(),
        0,
        std::max(maxCells, maxParticles) * 1.5,
        ImGuiCond_Always);
    static ImPlotAxisFlags flags = 0;
    ImPlot::BeginPlot(
        "",
        NULL,
        NULL,
        ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y * 0.5f),
        0,
        flags,
        flags);
    float labelPosY =
        _longtermStatistics.numCellsHistory.back() > _longtermStatistics.numParticlesHistory.back() ? -10.0f : 10.0f;

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

    ImPlot::EndPlot();

    ImPlot::SetNextPlotLimits(
        _longtermStatistics.timestepHistory.front(),
        _longtermStatistics.timestepHistory.back(),
        0,
        maxTokens * 1.5,
        ImGuiCond_Always);
    ImPlot::BeginPlot(
        "##Tokens",
        NULL,
        NULL,
        ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y),
        0,
        flags,
        flags);

    ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(2));
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
    ImPlot::PopStyleColor();

    ImPlot::EndPlot();
}

void _StatisticsWindow::updateData()
{
    auto monitorData = _simController->getMonitorData();

    //live statistics
    if (!_liveStatistics.timepointsHistory.empty() && _liveStatistics.timepointsHistory.back() - _liveStatistics.timepointsHistory.front() > (MaxLiveHistory + 1.0f)) {
        _liveStatistics.timepointsHistory.erase(_liveStatistics.timepointsHistory.begin());
        _liveStatistics.numCellsHistory.erase(_liveStatistics.numCellsHistory.begin());
        _liveStatistics.numParticlesHistory.erase(_liveStatistics.numParticlesHistory.begin());
        _liveStatistics.numTokensHistory.erase(_liveStatistics.numTokensHistory.begin());
    }

    _liveStatistics.timepoint += ImGui::GetIO().DeltaTime;
    _liveStatistics.timepointsHistory.emplace_back(_liveStatistics.timepoint);
    _liveStatistics.numCellsHistory.emplace_back(toFloat(monitorData.numCells));
    _liveStatistics.numParticlesHistory.emplace_back(toFloat(monitorData.numParticles));
    _liveStatistics.numTokensHistory.emplace_back(toFloat(monitorData.numTokens));

    //long-term statistics
    if (_longtermStatistics.timestepHistory.empty()
        || monitorData.timeStep - _longtermStatistics.timestepHistory.back() > LongtermTimestepDelta) {
        _longtermStatistics.timestepHistory.emplace_back(toFloat(monitorData.timeStep));
        _longtermStatistics.numCellsHistory.emplace_back(toFloat(monitorData.numCells));
        _longtermStatistics.numParticlesHistory.emplace_back(toFloat(monitorData.numParticles));
        _longtermStatistics.numTokensHistory.emplace_back(toFloat(monitorData.numTokens));
    }
}
