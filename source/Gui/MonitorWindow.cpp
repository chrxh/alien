#include "MonitorWindow.h"

#include "imgui.h"
#include "implot.h"

#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"

namespace
{
    float const MaxLiveHistory = 120.0f; //in seconds
}

_MonitorWindow::_MonitorWindow(SimulationController const& simController)
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

void _MonitorWindow::process()
{
    if (!_on) {
        return;
    }

    updateData();

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Monitor", &_on, windowFlags);

    ImGui::Checkbox("Live", &_live);

    ImGui::SameLine();
    ImGui::BeginDisabled(!_live);
    ImGui::SliderFloat("", &_history, 1, MaxLiveHistory, "%.1f s");
    ImGui::EndDisabled();
    
    auto maxCells = getMax(_numCellsHistory);
    auto maxParticles = getMax(_numParticlesHistory);
    auto maxTokens = getMax(_numTokensHistory);

    ImPlot::SetNextPlotLimits(
        _timepointsHistory.back() - _history,
        _timepointsHistory.back(),
        0,
        std::max(maxCells, maxParticles) * 1.5,
        ImGuiCond_Always);
    static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickLabels;
    ImPlot::BeginPlot("", NULL, NULL, ImVec2(-1, 150), 0, flags, flags);
    float labelPosY = _numCellsHistory.back() > _numParticlesHistory.back() ? -10.0f : 10.0f;

    ImPlot::PlotLine("Cells", _timepointsHistory.data(), _numCellsHistory.data(), toInt(_numCellsHistory.size()));
    ImPlot::AnnotateClamped(
        _timepointsHistory.back(),
        _numCellsHistory.back(),
        ImVec2(-10.0f, labelPosY),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_numCellsHistory.back())).c_str());

    ImPlot::PlotLine(
        "Energy particles", _timepointsHistory.data(), _numParticlesHistory.data(), toInt(_numParticlesHistory.size()));
    ImPlot::AnnotateClamped(
        _timepointsHistory.back(),
        _numParticlesHistory.back(),
        ImVec2(-10.0f, -labelPosY),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_numParticlesHistory.back())).c_str());

    ImPlot::EndPlot();

    ImPlot::SetNextPlotLimits(
        _timepointsHistory.back() - _history, _timepointsHistory.back(), 0, maxTokens * 1.5, ImGuiCond_Always);
    ImPlot::BeginPlot("##Tokens", NULL, NULL, ImVec2(-1, 150), 0, flags, flags);

    ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(2));
    ImPlot::PlotLine("Tokens", _timepointsHistory.data(), _numTokensHistory.data(), toInt(_numTokensHistory.size()));
    ImPlot::AnnotateClamped(
        _timepointsHistory.back(),
        _numTokensHistory.back(),
        ImVec2(-10, 10),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_numTokensHistory.back())).c_str());
    ImPlot::PopStyleColor();

    ImPlot::EndPlot();

    ImGui::End();
//    ImPlot::ShowDemoWindow();
}

void _MonitorWindow::updateData()
{
    auto monitorData = _simController->getMonitorData();

    if (!_timepointsHistory.empty() && _timepointsHistory.back() - _timepointsHistory.front() > (MaxLiveHistory + 1.0f)) {
        _timepointsHistory.erase(_timepointsHistory.begin());
        _numCellsHistory.erase(_numCellsHistory.begin());
        _numParticlesHistory.erase(_numParticlesHistory.begin());
        _numTokensHistory.erase(_numTokensHistory.begin());
    }

    _timepoint += ImGui::GetIO().DeltaTime;
    _timepointsHistory.emplace_back(_timepoint);
    _numCellsHistory.emplace_back(toFloat(monitorData.numCells));
    _numParticlesHistory.emplace_back(toFloat(monitorData.numParticles));
    _numTokensHistory.emplace_back(toFloat(monitorData.numTokens));
}
