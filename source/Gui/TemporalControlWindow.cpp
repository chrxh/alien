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
#include "OverlayMessageController.h"

namespace
{
    auto const LeftColumnWidth = 180.0f;
}

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
    AlienImGui::ToolbarSeparator();
    ImGui::SameLine();
    processStepBackwardButton();
    ImGui::SameLine();
    processStepForwardButton();
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();
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
    ImGui::SameLine(contentScale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    ImGui::BeginDisabled(!_slowDown);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderInt("", &_tpsRestriction, 1, 1000, "%d TPS", ImGuiSliderFlags_Logarithmic);
    if (_slowDown) {
        _simController->setTpsRestriction(_tpsRestriction);
    } else {
        _simController->setTpsRestriction(std::nullopt);
    }
    ImGui::PopItemWidth();
    ImGui::EndDisabled();

    auto syncSimulationWithRendering = _simController->isSyncSimulationWithRendering();
    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Sync with rendering"), syncSimulationWithRendering)) {
        _simController->setSyncSimulationWithRendering(syncSimulationWithRendering);
    }

    ImGui::BeginDisabled(!syncSimulationWithRendering);
    ImGui::SameLine(contentScale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    auto syncSimulationWithRenderingRatio = _simController->getSyncSimulationWithRenderingRatio();
    if (AlienImGui::SliderInt(AlienImGui::SliderIntParameters().textWidth(0).min(1).max(40).logarithmic(true).format("%d TPS : FPS"), &syncSimulationWithRenderingRatio)) {
        _simController->setSyncSimulationWithRenderingRatio(syncSimulationWithRenderingRatio);
    }
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
        printOverlayMessage("Snapshot taken");
    }
}

void _TemporalControlWindow::processRestoreButton()
{
    ImGui::BeginDisabled(!_snapshot);
    if (AlienImGui::ToolbarButton(ICON_FA_UNDO)) {
        _statisticsWindow->reset();
        _simController->setCurrentTimestep(_snapshot->timestep);
        _simController->setSimulationData(_snapshot->data);
        _simController->removeSelection();
        _history.clear();
        printOverlayMessage("Snapshot restored");   //flashback?
    }
    ImGui::EndDisabled();
}
