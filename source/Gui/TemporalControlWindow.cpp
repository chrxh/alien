#include "TemporalControlWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/Definitions.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SpaceCalculator.h"

#include "StyleRepository.h"
#include "StatisticsWindow.h"
#include "AlienImGui.h"
#include "DelayedExecutionController.h"
#include "OverlayController.h"

namespace
{
    auto constexpr LeftColumnWidth = 180.0f;
}

void TemporalControlWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void TemporalControlWindow::onSnapshot()
{
    _snapshot = createSnapshot();
}

TemporalControlWindow::TemporalControlWindow()
    : AlienWindow("Temporal control", "windows.temporal control", true)
{
}

void TemporalControlWindow::processIntern()
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
    processCreateFlashbackButton();
    ImGui::SameLine();
    processLoadFlashbackButton();

    AlienImGui::Separator();

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        processTpsInfo();
        processTotalTimestepsInfo();
        processRealTimeInfo();

        AlienImGui::Separator();
        processTpsRestriction();
    }
    ImGui::EndChild();
}

void TemporalControlWindow::processTpsInfo()
{
    ImGui::Text("Time steps per second");

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor /*0xffa07050*/);
    ImGui::TextUnformatted(StringHelper::format(_simulationFacade->getTps(), 1).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::Text("Total time steps");

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(_simulationFacade->getCurrentTimestep()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void TemporalControlWindow::processRealTimeInfo()
{
    ImGui::Text("Real-time");

    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(_simulationFacade->getRealTime()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void TemporalControlWindow::processTpsRestriction()
{
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Slow down"), _slowDown);
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    ImGui::BeginDisabled(!_slowDown);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderInt("##TPSRestriction", &_tpsRestriction, 1, 1000, "%d TPS", ImGuiSliderFlags_Logarithmic);
    if (_slowDown) {
        _simulationFacade->setTpsRestriction(_tpsRestriction);
    } else {
        _simulationFacade->setTpsRestriction(std::nullopt);
    }
    ImGui::PopItemWidth();
    ImGui::EndDisabled();

    auto syncSimulationWithRendering = _simulationFacade->isSyncSimulationWithRendering();
    if (AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Sync with rendering"), syncSimulationWithRendering)) {
        _simulationFacade->setSyncSimulationWithRendering(syncSimulationWithRendering);
    }

    ImGui::BeginDisabled(!syncSimulationWithRendering);
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    auto syncSimulationWithRenderingRatio = _simulationFacade->getSyncSimulationWithRenderingRatio();
    if (AlienImGui::SliderInt(AlienImGui::SliderIntParameters().textWidth(0).min(1).max(40).logarithmic(true).format("%d TPS : FPS"), &syncSimulationWithRenderingRatio)) {
        _simulationFacade->setSyncSimulationWithRenderingRatio(syncSimulationWithRenderingRatio);
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processRunButton()
{
    ImGui::BeginDisabled(_simulationFacade->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PLAY));
    AlienImGui::Tooltip("Run");
    if (result) {
        _history.clear();
        _simulationFacade->runSimulation();
        printOverlayMessage("Run");
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processPauseButton()
{
    ImGui::BeginDisabled(!_simulationFacade->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_PAUSE));
    AlienImGui::Tooltip("Pause");
    if (result) {
        _simulationFacade->pauseSimulation();
        printOverlayMessage("Pause");
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processStepBackwardButton()
{
    ImGui::BeginDisabled(_history.empty() || _simulationFacade->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_CHEVRON_LEFT));
    AlienImGui::Tooltip("Load previous time step");
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
    ImGui::BeginDisabled(_simulationFacade->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_CHEVRON_RIGHT));
    AlienImGui::Tooltip("Process single time step");
    if (result) {
        _history.emplace_back(createSnapshot());
        _simulationFacade->calcTimesteps(1);
    }
    ImGui::EndDisabled();
}

void TemporalControlWindow::processCreateFlashbackButton()
{
    auto result = AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_CAMERA));
    AlienImGui::Tooltip("Creating in-memory flashback: It saves the content of the current world to the memory.");
    if (result) {
        delayedExecution([this] { onSnapshot(); });
        
        printOverlayMessage("Creating flashback ...", true);
    }
}

void TemporalControlWindow::processLoadFlashbackButton()
{
    ImGui::BeginDisabled(!_snapshot);
    auto result = AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_UNDO));
    AlienImGui::Tooltip("Loading in-memory flashback: It loads the saved world from the memory. Static simulation parameters will not be changed. Non-static parameters "
                        "(such as the position of moving zones) will be restored as well.");
    if (result) {
        delayedExecution([this] { applySnapshot(*_snapshot); });
        _simulationFacade->removeSelection();
        _history.clear();

        printOverlayMessage("Loading flashback ...", true);
    }
    ImGui::EndDisabled();
}

TemporalControlWindow::Snapshot TemporalControlWindow::createSnapshot()
{
    Snapshot result;
    result.timestep = _simulationFacade->getCurrentTimestep();
    result.realTime = _simulationFacade->getRealTime();
    result.data = _simulationFacade->getSimulationData();
    result.parameters = _simulationFacade->getSimulationParameters();
    return result;
}


void TemporalControlWindow::applySnapshot(Snapshot const& snapshot)
{
    auto parameters = _simulationFacade->getSimulationParameters();
    auto const& origParameters = snapshot.parameters;

    if (origParameters.numRadiationSources == parameters.numRadiationSources) {
        for (int i = 0; i < parameters.numRadiationSources; ++i) {
            restorePosition(parameters.radiationSource[i], origParameters.radiationSource[i], snapshot.timestep);
        }
    }

    if (origParameters.numSpots == parameters.numSpots) {
        for (int i = 0; i < parameters.numSpots; ++i) {
            restorePosition(parameters.spot[i], origParameters.spot[i], snapshot.timestep);
        }
    }

    parameters.externalEnergy = origParameters.externalEnergy;
    if (parameters.cellMaxAgeBalancer || origParameters.cellMaxAgeBalancer) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.cellMaxAge[i] = origParameters.cellMaxAge[i];
        }
    }
    _simulationFacade->setCurrentTimestep(snapshot.timestep);
    _simulationFacade->setRealTime(snapshot.realTime);
    _simulationFacade->clear();
    _simulationFacade->setSimulationData(snapshot.data);
    _simulationFacade->setSimulationParameters(parameters);
}

template <typename MovedObjectType>
void TemporalControlWindow::restorePosition(MovedObjectType& movedObject, MovedObjectType const& origMovedObject, uint64_t origTimestep)
{
    auto origMovedObjectClone = origMovedObject;
    auto movedObjectClone = movedObject;

    if (std::abs(movedObject.velX) > NEAR_ZERO || std::abs(movedObject.velY) > NEAR_ZERO || std::abs(origMovedObject.velX) > NEAR_ZERO
        || std::abs(origMovedObject.velY) > NEAR_ZERO) {
        movedObject.posX = origMovedObject.posX;
        movedObject.posY = origMovedObject.posY;
    }
}
