#include "TemporalControlWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/Definitions.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"

#include "StyleRepository.h"
#include "StatisticsWindow.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"

_TemporalControlWindow::_TemporalControlWindow(
    SimulationController const& simController,
    StatisticsWindow const& statisticsWindow)
    : _AlienWindow("Temporal control", "windows.temporal control", true)
    , _simController(simController)
    , _statisticsWindow(statisticsWindow)
{}

void _TemporalControlWindow::onSnapshot()
{
    Snapshot newSnapshot;
    newSnapshot.timestep = _simController->getCurrentTimestep();
    newSnapshot.data = _simController->getSimulationData();
    _snapshot = newSnapshot;
}

void _TemporalControlWindow::processIntern()
{
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

        AlienImGui::Separator();
        processTpsRestriction();
    }
    ImGui::EndChild();
}

void _TemporalControlWindow::processTpsInfo()
{
    ImGui::Text("Time steps per second");

    ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor /*0xffa07050*/);
    ImGui::TextUnformatted(StringHelper::format(_simController->getTps(), 1).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::Text("Total time steps");

    ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(_simController->getCurrentTimestep()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTpsRestriction()
{
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Slow down"), _slowDown);
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
    if (AlienImGui::ToolbarButton(ICON_FA_PLAY)) {
        _history.clear();
        _simController->runSimulation();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processPauseButton()
{
    ImGui::BeginDisabled(!_simController->isSimulationRunning());
    if (AlienImGui::ToolbarButton(ICON_FA_PAUSE)) {
        _simController->pauseSimulation();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepBackwardButton()
{
    ImGui::BeginDisabled(_history.empty() || _simController->isSimulationRunning());
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_LEFT)) {
        auto const& snapshot = _history.back();
        _simController->setCurrentTimestep(snapshot.timestep);
        _simController->setSimulationData(snapshot.data);
        _history.pop_back();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepForwardButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    if (AlienImGui::ToolbarButton(ICON_FA_CHEVRON_RIGHT)) {
        Snapshot newSnapshot;
        newSnapshot.timestep = _simController->getCurrentTimestep();
        newSnapshot.data = _simController->getSimulationData();
        _history.emplace_back(newSnapshot);

        _simController->calcSingleTimestep();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processSnapshotButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_CAMERA)) {
        onSnapshot();
    }
}

void _TemporalControlWindow::processRestoreButton()
{
    ImGui::BeginDisabled(!_snapshot);
    if (AlienImGui::ToolbarButton(ICON_FA_UNDO)) {
        _statisticsWindow->reset();
        _simController->setCurrentTimestep(_snapshot->timestep);
        _simController->setSimulationData(_snapshot->data);
        _history.clear();
    }
    ImGui::EndDisabled();
}
