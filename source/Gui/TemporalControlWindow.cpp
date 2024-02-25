#include "TemporalControlWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/Definitions.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SpaceCalculator.h"

#include "StyleRepository.h"
#include "StatisticsWindow.h"
#include "AlienImGui.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"

namespace
{
    auto constexpr LeftColumnWidth = 180.0f;
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
    _snapshot = createSnapshot();
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
    processCreateFlashbackButton();
    ImGui::SameLine();
    processLoadFlashbackButton();

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
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
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
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    auto syncSimulationWithRenderingRatio = _simController->getSyncSimulationWithRenderingRatio();
    if (AlienImGui::SliderInt(AlienImGui::SliderIntParameters().textWidth(0).min(1).max(40).logarithmic(true).format("%d TPS : FPS"), &syncSimulationWithRenderingRatio)) {
        _simController->setSyncSimulationWithRenderingRatio(syncSimulationWithRenderingRatio);
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processRunButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(ICON_FA_PLAY);
    AlienImGui::Tooltip("Run");
    if (result) {
        _history.clear();
        _simController->runSimulation();
        printOverlayMessage("Run");
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processPauseButton()
{
    ImGui::BeginDisabled(!_simController->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(ICON_FA_PAUSE);
    AlienImGui::Tooltip("Pause");
    if (result) {
        _simController->pauseSimulation();
        printOverlayMessage("Pause");
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepBackwardButton()
{
    ImGui::BeginDisabled(_history.empty() || _simController->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(ICON_FA_CHEVRON_LEFT);
    AlienImGui::Tooltip("Load previous time step");
    if (result) {
        auto const& snapshot = _history.back();
        delayedExecution([this, snapshot] { applySnapshot(snapshot); });
        printOverlayMessage("Loading previous time step ...");

        _history.pop_back();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepForwardButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    auto result = AlienImGui::ToolbarButton(ICON_FA_CHEVRON_RIGHT);
    AlienImGui::Tooltip("Process single time step");
    if (result) {
        _history.emplace_back(createSnapshot());
        _simController->calcTimesteps(1);
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processCreateFlashbackButton()
{
    auto result = AlienImGui::ToolbarButton(ICON_FA_CAMERA);
    AlienImGui::Tooltip("Create flashback");
    if (result) {
        delayedExecution([this] { onSnapshot(); });
        
        printOverlayMessage("Creating flashback ...", true);
    }
}

void _TemporalControlWindow::processLoadFlashbackButton()
{
    ImGui::BeginDisabled(!_snapshot);
    auto result = AlienImGui::ToolbarButton(ICON_FA_UNDO);
    AlienImGui::Tooltip("Load flashback");
    if (result) {
        delayedExecution([this] { applySnapshot(*_snapshot); });
        _simController->removeSelection();
        _history.clear();

        printOverlayMessage("Loading flashback ...", true);
    }
    ImGui::EndDisabled();
}

_TemporalControlWindow::Snapshot _TemporalControlWindow::createSnapshot()
{
    Snapshot result;
    result.timestep = _simController->getCurrentTimestep();
    result.data = _simController->getSimulationData();
    result.parameters = _simController->getSimulationParameters();
    return result;
}


void _TemporalControlWindow::applySnapshot(Snapshot const& snapshot)
{
    auto parameters = _simController->getSimulationParameters();
    auto const& origParameters = snapshot.parameters;

    if (origParameters.numParticleSources == parameters.numParticleSources) {
        for (int i = 0; i < parameters.numParticleSources; ++i) {
            restorePosition(parameters.particleSources[i], origParameters.particleSources[i], snapshot.timestep);
        }
    }

    if (origParameters.numSpots == parameters.numSpots) {
        for (int i = 0; i < parameters.numSpots; ++i) {
            restorePosition(parameters.spots[i], origParameters.spots[i], snapshot.timestep);
        }
    }

    for (int i = 0; i < MAX_COLORS; ++i) {
        parameters.cellFunctionConstructorExternalEnergy[i] = origParameters.cellFunctionConstructorExternalEnergy[i];
    }
    if (parameters.cellMaxAgeBalancer || origParameters.cellMaxAgeBalancer) {
        for (int i = 0; i < MAX_COLORS; ++i) {
            parameters.cellMaxAge[i] = origParameters.cellMaxAge[i];
        }
    }
    _simController->setCurrentTimestep(snapshot.timestep);
    _simController->clear();
    _simController->setSimulationData(snapshot.data);
    _simController->setSimulationParameters(parameters);
}

template <typename MovedObjectType>
void _TemporalControlWindow::restorePosition(MovedObjectType& movedObject, MovedObjectType const& origMovedObject, uint64_t origTimestep)
{
    auto origMovedObjectClone = origMovedObject;
    auto movedObjectClone = movedObject;

    if (std::abs(movedObject.velX) > NEAR_ZERO || std::abs(movedObject.velY) > NEAR_ZERO || std::abs(origMovedObject.velX) > NEAR_ZERO
        || std::abs(origMovedObject.velY) > NEAR_ZERO) {
        movedObject.posX = origMovedObject.posX;
        movedObject.posY = origMovedObject.posY;
    }
}
