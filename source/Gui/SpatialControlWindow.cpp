#include "SpatialControlWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationFacade.h"

#include "StyleRepository.h"
#include "Viewport.h"
#include "AlienImGui.h"
#include "ResizeWorldDialog.h"

void SpatialControlWindow::init(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;
    _resizeWorldDialog = std::make_shared<_ResizeWorldDialog>(simulationFacade);

    auto& settings = GlobalSettings::get();
    Viewport::get().setZoomSensitivity(settings.getFloat("windows.spatial control.zoom sensitivity factor", Viewport::get().getZoomSensitivity()));
}

SpatialControlWindow::SpatialControlWindow()
    : AlienWindow("Spatial control", "windows.spatial control", true)
{}

void SpatialControlWindow::shutdownIntern()
{
    auto& settings = GlobalSettings::get();
    settings.setFloat("windows.spatial control.zoom sensitivity", Viewport::get().getZoomSensitivity());
}

void SpatialControlWindow::processIntern()
{
    processZoomInButton();
    ImGui::SameLine();
    processZoomOutButton();
    ImGui::SameLine();
    processCenterButton();
    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();
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
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        auto worldSize = _simulationFacade->getWorldSize();
        ImGui::TextUnformatted(
            (StringHelper::format(worldSize.x) + " x " + StringHelper::format(worldSize.y)).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Zoom factor");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(StringHelper::format(Viewport::get().getZoomFactor(), 2).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Center position");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        auto centerPos = Viewport::get().getCenterInWorldPos();
        ImGui::TextUnformatted(
            (StringHelper::format(centerPos.x, 1) + ", " + StringHelper::format(centerPos.y, 1)).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        AlienImGui::Separator();
        AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Autotracking on selection"), _centerSelection);
        ImGui::Spacing();
        ImGui::Spacing();
        float sensitivity = Viewport::get().getZoomSensitivity();
        if (AlienImGui::SliderFloat(AlienImGui::SliderFloatParameters().name("Zoom sensitivity").min(1.0f).max(1.1f).textWidth(130).format(""), &sensitivity)) {
            Viewport::get().setZoomSensitivity(sensitivity);
        }
    }
    ImGui::EndChild();

    _resizeWorldDialog->process();
}

void SpatialControlWindow::processBackground()
{
    processCenterOnSelection();
}

void SpatialControlWindow::processZoomInButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_SEARCH_PLUS)) {
        Viewport::get().setZoomFactor(Viewport::get().getZoomFactor() * 2);
    }
    AlienImGui::Tooltip("Zoom in");
}

void SpatialControlWindow::processZoomOutButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_SEARCH_MINUS)) {
        Viewport::get().setZoomFactor(Viewport::get().getZoomFactor() / 2);
    }
    AlienImGui::Tooltip("Zoom out");
}

void SpatialControlWindow::processCenterButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_CROSSHAIRS)) {
        Viewport::get().setZoomFactor(1.0f);
        auto worldSize = toRealVector2D(_simulationFacade->getWorldSize());
        Viewport::get().setCenterInWorldPos({worldSize.x / 2, worldSize.y / 2});
    }
    AlienImGui::Tooltip("Center");
}

void SpatialControlWindow::processResizeButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_CROP_ALT)) {
        _resizeWorldDialog->open();
    }
    AlienImGui::Tooltip("Resize");
}

void SpatialControlWindow::processCenterOnSelection()
{
    if (_centerSelection && _simulationFacade->isSimulationRunning()) {
        auto shallowData = _simulationFacade->getSelectionShallowData();
        if (shallowData.numCells > 0 || shallowData.numParticles > 0) {
            Viewport::get().setCenterInWorldPos({shallowData.centerPosX, shallowData.centerPosY});
        }
    }
}
