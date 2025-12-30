#include "TemporalControlWindow.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SpaceCalculator.h>

#include "AlienGui.h"
#include "DelayedExecutionController.h"
#include "OverlayController.h"
#include "StatisticsWindow.h"
#include "StyleRepository.h"
#include <EngineInterface/SimulationFacade.h>

namespace
{
    auto constexpr LeftColumnWidth = 180.0f;
}

void TemporalControlWindow::initIntern()
{

}

void TemporalControlWindow::onSnapshot()
{
    _snapshot = createSnapshot();
}

TemporalControlWindow::TemporalControlWindow()
    : AlienWindow("Temporal control", "windows.temporal control", true)
{}

void TemporalControlWindow::processIntern()
{
    processRunButton();
    ImGui::SameLine();
    processPauseButton();
    ImGui::SameLine();
    AlienGui::ToolbarSeparator();
    ImGui::SameLine();
    processStepBackwardButton();
    ImGui::SameLine();
    processStepForwardButton();
    ImGui::SameLine();
    AlienGui::ToolbarSeparator();
    ImGui::SameLine();
    processCreateFlashbackButton();
    ImGui::SameLine();
    processLoadFlashbackButton();

    AlienGui::Separator();

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        processTpsInfo();
        processTotalTimestepsInfo();
        processRealTimeInfo();

        AlienGui::Separator();
        processTpsRestriction();
    }
    ImGui::EndChild();

    if (!_sessionId.has_value() || _sessionId.value() != _SimulationFacade::get()->getSessionId()) {
        _history.clear();
    }
    _sessionId = _SimulationFacade::get()->getSessionId();
}

void TemporalControlWindow::processTpsInfo()
{
    ImGui::Text("Time steps per second");

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value /*0xffa07050*/);
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getTps(), 1).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::Text("Total time steps");

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getCurrentTimestep()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void TemporalControlWindow::processRealTimeInfo()
{
    ImGui::Text("Real-time");

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getRealTime()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void TemporalControlWindow::processTpsRestriction()
{
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Slow down"), _slowDown);
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    ImGui::BeginDisabled(!_slowDown);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderInt("##TPSRestriction", &_tpsRestriction, 1, 1000, "%d TPS", ImGuiSliderFlags_Logarithmic);
    if (_slowDown) {
        _SimulationFacade::get()->setTpsRestriction(_tpsRestriction);
    } else {
        _SimulationFacade::get()->setTpsRestriction(std::nullopt);
    }
    ImGui::PopItemWidth();
    ImGui::EndDisabled();

    auto syncSimulationWithRendering = _SimulationFacade::get()->isSyncSimulationWithRendering();
    if (AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Sync with rendering"), syncSimulationWithRendering)) {
        _SimulationFacade::get()->setSyncSimulationWithRendering(syncSimulationWithRendering);
    }

    ImGui::BeginDisabled(!syncSimulationWithRendering);
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    auto syncSimulationWithRenderingRatio = _SimulationFacade::get()->getSyncSimulationWithRenderingRatio();
    if (AlienGui::SliderInt(
            AlienGui::SliderIntParameters().textWidth(0).min(1).max(40).logarithmic(true).format("%d TPS : FPS"), &syncSimulationWithRenderingRatio)) {
        _SimulationFacade::get()->setSyncSimulationWithRenderingRatio(syncSimulationWithRenderingRatio);
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processRunButton()
{
    ImGui::BeginDisabled(_SimulationFacade::get()->isSimulationRunning());
    auto result = AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_PLAY));
    AlienGui::Tooltip("Run");
    if (result) {
        _history.clear();
        _SimulationFacade::get()->runSimulation();
        printOverlayMessage("Run");
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processPauseButton()
{
    ImGui::BeginDisabled(!_SimulationFacade::get()->isSimulationRunning());
    auto result = AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_PAUSE));
    AlienGui::Tooltip("Pause");
    if (result) {
        _SimulationFacade::get()->pauseSimulation();
        printOverlayMessage("Pause");
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processStepBackwardButton()
{
    ImGui::BeginDisabled(_history.empty() || _SimulationFacade::get()->isSimulationRunning());
    auto result = AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CHEVRON_LEFT));
    AlienGui::Tooltip("Load previous time step");
    if (result) {
        auto const& snapshot = _history.back();
        delayedExecution([this, snapshot] { applySnapshot(snapshot); });
        printOverlayMessage("Loading previous time step ...");

        _history.pop_back();
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processStepForwardButton()
{
    ImGui::BeginDisabled(_SimulationFacade::get()->isSimulationRunning());
    auto result = AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CHEVRON_RIGHT));
    AlienGui::Tooltip("Process single time step");
    if (result) {
        _history.emplace_back(createSnapshot());
        _SimulationFacade::get()->calcTimesteps(1);
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processCreateFlashbackButton()
{
    auto result = AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CAMERA));
    AlienGui::Tooltip("Creating in-memory flashback: It saves the content of the current world to the memory.");
    if (result) {
        delayedExecution([this] { onSnapshot(); });

        printOverlayMessage("Creating flashback ...", true);
    }
}

void TemporalControlWindow::processLoadFlashbackButton()
{
    ImGui::BeginDisabled(!_snapshot);
    auto result = AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_UNDO));
    AlienGui::Tooltip(
        "Loading in-memory flashback: It loads the saved world from the memory. Static simulation parameters will not be changed. Non-static parameters "
        "(such as the position of moving layers) will be restored as well.");
    if (result) {
        delayedExecution([this] { applySnapshot(*_snapshot); });
        _SimulationFacade::get()->removeSelection();
        _history.clear();

        printOverlayMessage("Loading flashback ...", true);
    }
    ImGui::EndDisabled();
}

TemporalControlWindow::Snapshot TemporalControlWindow::createSnapshot()
{
    Snapshot result;
    result.timestep = _SimulationFacade::get()->getCurrentTimestep();
    result.realTime = _SimulationFacade::get()->getRealTime();
    result.data = _SimulationFacade::get()->getSimulationData();
    result.parameters = _SimulationFacade::get()->getSimulationParameters();
    return result;
}


void TemporalControlWindow::applySnapshot(Snapshot const& snapshot)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto const& origParameters = snapshot.parameters;

    if (origParameters.numLayers == parameters.numLayers) {
        for (int i = 0; i < parameters.numLayers; ++i) {
            restorePosition(
                parameters.layerPosition.layerValues[i],
                parameters.layerVelocity.layerValues[i],
                origParameters.layerPosition.layerValues[i],
                origParameters.layerVelocity.layerValues[i]);
        }
    }

    if (origParameters.numSources == parameters.numSources) {
        for (int i = 0; i < parameters.numLayers; ++i) {
            restorePosition(
                parameters.sourcePosition.sourceValues[i],
                parameters.sourceVelocity.sourceValues[i],
                origParameters.sourcePosition.sourceValues[i],
                origParameters.sourceVelocity.sourceValues[i]);
        }
    }

    parameters.externalEnergy = origParameters.externalEnergy;
    if (parameters.maxCellAgeBalancerInterval.enabled || origParameters.maxCellAgeBalancerInterval.enabled) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.maxCellAge.value[i] = origParameters.maxCellAge.value[i];
        }
    }
    auto simRunning = _SimulationFacade::get()->isSimulationRunning();
    if (simRunning) {
        _SimulationFacade::get()->pauseSimulation();
    }
    _SimulationFacade::get()->setCurrentTimestep(snapshot.timestep);
    _SimulationFacade::get()->setRealTime(snapshot.realTime);
    _SimulationFacade::get()->clear();
    _SimulationFacade::get()->setSimulationData(snapshot.data);
    _SimulationFacade::get()->setSimulationParameters(parameters);
    if (simRunning) {
        _SimulationFacade::get()->runSimulation();
    }
}

void TemporalControlWindow::restorePosition(
    RealVector2D& position,
    RealVector2D const& velocity,
    RealVector2D const& origPosition,
    RealVector2D const& origVelocity)
{
    if (std::abs(velocity.x) > NEAR_ZERO || std::abs(velocity.y) > NEAR_ZERO || std::abs(origVelocity.x) > NEAR_ZERO || std::abs(origVelocity.y) > NEAR_ZERO) {
        position = origPosition;
    }
}
