#include "TemporalControlWindow.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include <Data/SpaceCalculator.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/TemporalControlService.h>

#include "AlienGui.h"
#include "DelayedExecutionController.h"
#include "OverlayController.h"
#include "StyleService.h"

namespace
{
    auto constexpr LeftColumnWidth = 180.0f;
}

void TemporalControlWindow::initIntern() {}

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
}

void TemporalControlWindow::processTpsInfo()
{
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::Text("Time steps per second");
    ImGui::PopStyleColor();

    ImGui::PushFont(StyleService::get().getLargeFont());
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getTps(), 1).c_str());
    ImGui::PopFont();
}

void TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::Text("Total time steps");
    ImGui::PopStyleColor();

    ImGui::PushFont(StyleService::get().getLargeFont());
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getCurrentTimestep()).c_str());
    ImGui::PopFont();
}

void TemporalControlWindow::processRealTimeInfo()
{
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
    ImGui::Text("Real-time");
    ImGui::PopStyleColor();

    ImGui::PushFont(StyleService::get().getLargeFont());
    ImGui::TextUnformatted(StringHelper::format(_SimulationFacade::get()->getRealTime()).c_str());
    ImGui::PopFont();
}

void TemporalControlWindow::processTpsRestriction()
{
    auto tpsRestriction = _SimulationFacade::get()->getTpsRestriction();
    if (tpsRestriction) {
        _tpsRestriction = *tpsRestriction;
    }
    auto slowDown = tpsRestriction.has_value();
    if (AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Slow down"), slowDown)) {
        _SimulationFacade::get()->setTpsRestriction(slowDown ? std::make_optional(_tpsRestriction) : std::nullopt);
    }
    ImGui::SameLine(scale(LeftColumnWidth) - (ImGui::GetWindowWidth() - ImGui::GetContentRegionAvail().x));
    ImGui::BeginDisabled(!slowDown);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::SliderInt("##TPSRestriction", &_tpsRestriction, 1, 1000, "%d TPS", ImGuiSliderFlags_Logarithmic) && slowDown) {
        _SimulationFacade::get()->setTpsRestriction(_tpsRestriction);
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
    auto& temporalControlService = TemporalControlService::get();
    AlienGui::Toolbar(
        AlienGui::ToolbarParameters().id("TemporalControl"),
        {AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(ICON_FA_PLAY).name("Run").disabled(simulationRunning).action([&] {
             temporalControlService.clearPreviousTimesteps();
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
                                                 .disabled(!temporalControlService.hasPreviousTimestep() || simulationRunning)
                                                 .action([&] {
                                                     delayedExecution([] { TemporalControlService::get().restorePreviousTimestep(); });
                                                     printOverlayMessage("Loading previous time step ...");
                                                 })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters().icon(ICON_FA_CHEVRON_RIGHT).name("Process single time step").disabled(simulationRunning).action([&] {
                 temporalControlService.calcSingleTimestep();
             })),
         AlienGui::ToolbarItem::createSeparator(),
         AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                 .icon(ICON_FA_CAMERA)
                                                 .name("Create flashback")
                                                 .tooltip("Creating in-memory flashback: It saves the content of the current world to the memory.")
                                                 .action([&] {
                                                     delayedExecution([] { TemporalControlService::get().createFlashback(); });

                                                     printOverlayMessage("Creating flashback ...", true);
                                                 })),
         AlienGui::ToolbarItem::createButton(
             AlienGui::ToolbarItemParameters()
                 .icon(ICON_FA_UNDO)
                 .name("Load flashback")
                 .tooltip("Loading in-memory flashback: It loads the saved world from the memory. Static simulation parameters will not be changed. "
                          "Non-static parameters (such as the position of moving layers) will be restored as well.")
                 .disabled(!temporalControlService.hasFlashback())
                 .action([&] {
                     delayedExecution([] { TemporalControlService::get().restoreFlashback(); });

                     printOverlayMessage("Loading flashback ...", true);
                 }))});
}
