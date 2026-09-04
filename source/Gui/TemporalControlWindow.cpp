#include "TemporalControlWindow.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SpaceCalculator.h>

#include <EngineInterface/SimulationFacade.h>
#include "AlienGui.h"
#include "DelayedExecutionController.h"
#include "OverlayController.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr LeftColumnWidth = 180.0f;
}

void TemporalControlWindow::initIntern() {}

void TemporalControlWindow::onSnapshot()
{
    _snapshot = createSnapshot();
}

TemporalControlWindow::TemporalControlWindow()
    : AlienWindow("Temporal control", "windows.temporal control", true, false, {1517.0f, 578.0f}, {341.0f, 393.0f})
{}

void TemporalControlWindow::processIntern()
{
    processToolbar();

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
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::Text("Time steps per second");
    ImGui::PopStyleColor();

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getTps(), 1).c_str());
    ImGui::PopFont();
}

void TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::Text("Total time steps");
    ImGui::PopStyleColor();

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getCurrentTimestep()).c_str());
    ImGui::PopFont();
}

void TemporalControlWindow::processRealTimeInfo()
{
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::Text("Real-time");
    ImGui::PopStyleColor();

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getRealTime()).c_str());
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

void TemporalControlWindow::processToolbar()
{
    auto simulationRunning = _SimulationFacade::get()->isSimulationRunning();
    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("TemporalControl"),
        {AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(ICON_FA_PLAY).name("Run").disabled(simulationRunning).action([&] {
             _history.clear();
             _SimulationFacade::get()->runSimulation();
             printOverlayMessage("Run");
         })),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(ICON_FA_PAUSE).name("Pause").disabled(!simulationRunning).action([&] {
             _SimulationFacade::get()->pauseSimulation();
             printOverlayMessage("Pause");
         })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_CHEVRON_LEFT)
                                                 .name("Load previous time step")
                                                 .disabled(_history.empty() || simulationRunning)
                                                 .action([&] {
                                                     auto const& snapshot = _history.back();
                                                     delayedExecution([this, snapshot] { applySnapshot(snapshot); });
                                                     printOverlayMessage("Loading previous time step ...");

                                                     _history.pop_back();
                                                 })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_CHEVRON_RIGHT).name("Process single time step").disabled(simulationRunning).action([&] {
                 _history.emplace_back(createSnapshot());
                 _SimulationFacade::get()->calcTimesteps(1);
             })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_CAMERA)
                                                 .name("Create flashback")
                                                 .tooltip("Creating in-memory flashback: It saves the content of the current world to the memory.")
                                                 .action([&] {
                                                     delayedExecution([this] { onSnapshot(); });

                                                     printOverlayMessage("Creating flashback ...", true);
                                                 })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters()
                 .icon(ICON_FA_UNDO)
                 .name("Load flashback")
                 .tooltip("Loading in-memory flashback: It loads the saved world from the memory. Static simulation parameters will not be changed. "
                          "Non-static parameters (such as the position of moving layers) will be restored as well.")
                 .disabled(!_snapshot)
                 .action([&] {
                     delayedExecution([this] { applySnapshot(*_snapshot); });
                     _SimulationFacade::get()->removeSelection();
                     _history.clear();

                     printOverlayMessage("Loading flashback ...", true);
                 }))});
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
