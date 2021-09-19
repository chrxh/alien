#include "MonitorWindow.h"

#include "imgui.h"
#include "implot.h"

#include "EngineImpl/SimulationController.h"

_MonitorWindow::_MonitorWindow(SimulationController const& simController)
    : _simController(simController)
{
    _timestepsHistory.reserve(1000);
    _numCellsHistory.reserve(1000);
    _numParticlesHistory.reserve(1000);
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

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None;

    
    updateData();

    ImGui::Begin("Monitor", &_on, windowFlags);

    auto maxCells = getMax(_numCellsHistory);
    auto maxParticles = getMax(_numParticlesHistory);
    auto maxTokens = getMax(_numTokensHistory);

    ImPlot::SetNextPlotLimits(
        _timestepsHistory.front(), _timestepsHistory.back(), 0, std::max(maxCells, maxParticles) * 1.5, ImGuiCond_Always);
    static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickLabels;
    ImPlot::BeginPlot("##CellsParticles", NULL, NULL, ImVec2(-1, 150), 0, flags, flags);

    ImPlot::PlotLine("Cells", _timestepsHistory.data(), _numCellsHistory.data(), toInt(_timestepsHistory.size()));
    float labelPosY = _numCellsHistory.back() > _numParticlesHistory.back() ? -10 : 10;
    ImPlot::AnnotateClamped(
        _timestepsHistory.back(),
        _numCellsHistory.back(),
        ImVec2(-10, labelPosY),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_numCellsHistory.back())).c_str());

    ImPlot::PlotLine(
        "Energy particles", _timestepsHistory.data(), _numParticlesHistory.data(), toInt(_timestepsHistory.size()));
    ImPlot::AnnotateClamped(
        _timestepsHistory.back(),
        _numParticlesHistory.back(),
        ImVec2(-10, -labelPosY),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_numParticlesHistory.back())).c_str());

    ImPlot::EndPlot();


    ImPlot::SetNextPlotLimits(
        _timestepsHistory.front(), _timestepsHistory.back(), 0, maxTokens * 1.5, ImGuiCond_Always);
    ImPlot::BeginPlot("##Tokens", NULL, NULL, ImVec2(-1, 150), 0, flags, flags);

    ImPlot::PushStyleColor(ImPlotCol_Line, ImPlot::GetColormapColor(2));
    ImPlot::PlotLine(
        "Tokens", _timestepsHistory.data(), _numTokensHistory.data(), toInt(_timestepsHistory.size()));
    ImPlot::AnnotateClamped(
        _timestepsHistory.back(),
        _numTokensHistory.back(),
        ImVec2(-10, 10),
        ImPlot::GetLastItemColor(),
        std::to_string(toInt(_numTokensHistory.back())).c_str());
    ImPlot::PopStyleColor();

    ImPlot::EndPlot();

    ImGui::End();
    ImPlot::ShowDemoWindow();

/*
   float maxCells = getMax(_numCellsHistory);
    ImGui::PlotLines(
        "Cells",
        _numCellsHistory.data(),
        _numCellsHistory.size(),
        0,
        std::to_string(_numCellsHistory[0]).c_str(),
        0.0f,
        maxCells,
        ImVec2(0, 80.0f));
*/
}

void _MonitorWindow::updateData()
{
    if (_numCellsHistory.size() > 1000) {
        _timestepsHistory.erase(_timestepsHistory.begin());
        _numCellsHistory.erase(_numCellsHistory.begin());
        _numParticlesHistory.erase(_numParticlesHistory.begin());
        _numTokensHistory.erase(_numTokensHistory.begin());
    }

    auto monitorData = _simController->getMonitorData();

    _timestepsHistory.emplace_back(toFloat(monitorData.timeStep));
    _numCellsHistory.emplace_back(toFloat(monitorData.numCells));
    _numParticlesHistory.emplace_back(toFloat(monitorData.numParticles));
    _numTokensHistory.emplace_back(toFloat(monitorData.numTokens));
}
