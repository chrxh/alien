#include "TemporalControlWindow.h"

#include <imgui.h>

#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "Base/Definitions.h"
#include "Base/StringFormatter.h"
#include "EngineImpl/SimulationController.h"

#include "StyleRepository.h"
#include "Resources.h"
#include "StatisticsWindow.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"

_TemporalControlWindow::_TemporalControlWindow(
    SimulationController const& simController,
    StatisticsWindow const& statisticsWindow)
    : _simController(simController)
    , _statisticsWindow(statisticsWindow)
{
    _on = GlobalSettings::getInstance().getBoolState("windows.temporal control.active", true);
}

_TemporalControlWindow::~_TemporalControlWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.temporal control.active", _on);
}

void _TemporalControlWindow::process()
{
    if (!_on) {
        return;
    }

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::Begin("Temporal control", &_on);

    processRunButton();
    ImGui::SameLine();
    processPauseButton();
    ImGui::SameLine();
    processStepBackwardButton();
    ImGui::SameLine();
    processStepForwardButton();
    ImGui::SameLine();
    processSnapshotButton();
    ImGui::SameLine();
    processRestoreButton();

    AlienImGui::Separator();

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        processTpsInfo();
        processTotalTimestepsInfo();

        ImGui::Spacing();
        ImGui::Spacing();
        processTpsRestriction();
    }
    ImGui::EndChild();

    ImGui::End();
}

bool _TemporalControlWindow::isOn() const
{
    return _on;
}

void _TemporalControlWindow::setOn(bool value)
{
    _on = value;
}

void _TemporalControlWindow::processTpsInfo()
{
    ImGui::Text("Time steps per second");

    ImGui::PushFont(StyleRepository::getInstance().getHugeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor /*0xffa07050*/);
    ImGui::TextUnformatted(StringFormatter::format(_simController->getTps(), 1).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::Text("Total time steps");

    ImGui::PushFont(StyleRepository::getInstance().getHugeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringFormatter::format(_simController->getCurrentTimestep()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTpsRestriction()
{
    ImGui::Checkbox("Slow down", &_slowDown);
    ImGui::SameLine();
    ImGui::BeginDisabled(!_slowDown);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderInt("", &_tpsRestriction, 1, 400, "%d TPS", ImGuiSliderFlags_Logarithmic);
    if (_slowDown) {
        _simController->setTpsRestriction(_tpsRestriction);
    } else {
        _simController->setTpsRestriction(std::nullopt);
    }
    ImGui::PopItemWidth();
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processRunButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    if (AlienImGui::BeginToolbarButton(ICON_FA_PLAY)) {
        _history.clear();
        _simController->runSimulation();
    }
    AlienImGui::EndToolbarButton();
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processPauseButton()
{
    ImGui::BeginDisabled(!_simController->isSimulationRunning());
    if (AlienImGui::BeginToolbarButton(ICON_FA_PAUSE)) {
        _simController->pauseSimulation();
    }
    AlienImGui::EndToolbarButton();
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepBackwardButton()
{
    ImGui::BeginDisabled(_history.empty() || _simController->isSimulationRunning());
    if (AlienImGui::BeginToolbarButton(ICON_FA_CHEVRON_LEFT)) {
        auto const& snapshot = _history.back();
        _simController->setCurrentTimestep(snapshot.timestep);
        _simController->setSimulationData(snapshot.data);
        _history.pop_back();
    }
    AlienImGui::EndToolbarButton();
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepForwardButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    if (AlienImGui::BeginToolbarButton(ICON_FA_CHEVRON_RIGHT)) {
        Snapshot newSnapshot;
        newSnapshot.timestep = _simController->getCurrentTimestep();
        auto size = _simController->getWorldSize();
        newSnapshot.data = _simController->getSimulationData({0, 0}, size);
        _history.emplace_back(newSnapshot);

        _simController->calcSingleTimestep();
    }
    AlienImGui::EndToolbarButton();
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processSnapshotButton()
{
    if (AlienImGui::BeginToolbarButton(ICON_FA_CAMERA)) {
        Snapshot newSnapshot;
        newSnapshot.timestep = _simController->getCurrentTimestep();
        auto size = _simController->getWorldSize();
        newSnapshot.data = _simController->getSimulationData({0, 0}, size);
        _snapshot = newSnapshot;
    }
    AlienImGui::EndToolbarButton();
}

void _TemporalControlWindow::processRestoreButton()
{
    ImGui::BeginDisabled(!_snapshot);
    if (AlienImGui::BeginToolbarButton(ICON_FA_UNDO)) {
        _statisticsWindow->reset();
        _simController->setCurrentTimestep(_snapshot->timestep);
        _simController->setSimulationData(_snapshot->data);
    }
    AlienImGui::EndToolbarButton();
    ImGui::EndDisabled();
}
