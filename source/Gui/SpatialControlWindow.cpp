#include "SpatialControlWindow.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "ResizeWorldDialog.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "SimulationFacadeProvider.h"

void SpatialControlWindow::initIntern()
{

    ResizeWorldDialog::get().setup();

    auto& settings = GlobalSettings::get();
    Viewport::get().setZoomSensitivity(settings.getValue("windows.spatial control.zoom sensitivity factor", Viewport::get().getZoomSensitivity()));
}

SpatialControlWindow::SpatialControlWindow()
    : AlienWindow("Spatial control", "windows.spatial control", true)
{}

void SpatialControlWindow::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    settings.setValue("windows.spatial control.zoom sensitivity", Viewport::get().getZoomSensitivity());
}

void SpatialControlWindow::processIntern()
{
    processZoomInButton();
    ImGui::SameLine();
    processZoomOutButton();
    ImGui::SameLine();
    processCenterButton();
    ImGui::SameLine();
    AlienGui::ToolbarSeparator();
    ImGui::SameLine();
    processResizeButton();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::Text("World size");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        auto worldSize = SimulationFacadeProvider::getSimulationFacade()->getWorldSize();
        ImGui::TextUnformatted((StringHelper::format(worldSize.x) + " x " + StringHelper::format(worldSize.y)).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Zoom factor");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        ImGui::TextUnformatted(StringHelper::format(Viewport::get().getZoomFactor(), 2).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Center position");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        auto centerPos = Viewport::get().getCenterInWorldPos();
        ImGui::TextUnformatted((StringHelper::format(centerPos.x, 1) + ", " + StringHelper::format(centerPos.y, 1)).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        AlienGui::Separator();
        AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Autotracking on selection"), _centerSelection);
        ImGui::Spacing();
        ImGui::Spacing();
        float sensitivity = Viewport::get().getZoomSensitivity();
        if (AlienGui::SliderFloat(AlienGui::SliderFloatParameters().name("Zoom sensitivity").min(1.0f).max(1.1f).textWidth(130).format(""), &sensitivity)) {
            Viewport::get().setZoomSensitivity(sensitivity);
        }
    }
    ImGui::EndChild();
}

void SpatialControlWindow::processBackground()
{
    processCenterOnSelection();
}

void SpatialControlWindow::processZoomInButton()
{
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_SEARCH_PLUS))) {
        Viewport::get().setZoomFactor(Viewport::get().getZoomFactor() * 2);
    }
    AlienGui::Tooltip("Zoom in");
}

void SpatialControlWindow::processZoomOutButton()
{
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_SEARCH_MINUS))) {
        Viewport::get().setZoomFactor(Viewport::get().getZoomFactor() / 2);
    }
    AlienGui::Tooltip("Zoom out");
}

void SpatialControlWindow::processCenterButton()
{
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CROSSHAIRS))) {
        Viewport::get().setZoomFactor(1.0f);
        auto worldSize = toRealVector2D(SimulationFacadeProvider::getSimulationFacade()->getWorldSize());
        Viewport::get().setCenterInWorldPos({worldSize.x / 2, worldSize.y / 2});
    }
    AlienGui::Tooltip("Center");
}

void SpatialControlWindow::processResizeButton()
{
    if (AlienGui::ToolbarButton(AlienGui::ToolbarButtonParameters().text(ICON_FA_CROP_ALT))) {
        ResizeWorldDialog::get().open();
    }
    AlienGui::Tooltip("Resize");
}

void SpatialControlWindow::processCenterOnSelection()
{
    if (_centerSelection && SimulationFacadeProvider::getSimulationFacade()->isSimulationRunning()) {
        auto shallowData = SimulationFacadeProvider::getSimulationFacade()->getSelectionShallowData();
        if (shallowData.numCells > 0 || shallowData.numParticles > 0) {
            Viewport::get().setCenterInWorldPos({shallowData.centerPosX, shallowData.centerPosY});
        }
    }
}
