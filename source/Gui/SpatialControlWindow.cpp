#include "SpatialControlWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/SimulationController.h"

#include "StyleRepository.h"
#include "Viewport.h"
#include "AlienImGui.h"
#include "ResizeWorldDialog.h"

_SpatialControlWindow::_SpatialControlWindow(
    SimulationController const& simController,
    TemporalControlWindow const& temporalControlWindow)
    : _AlienWindow("Spatial control", "windows.spatial control", true)
    , _simController(simController)
{
    _resizeWorldDialog = std::make_shared<_ResizeWorldDialog>(simController, temporalControlWindow);
}

void _SpatialControlWindow::processIntern()
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
        ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        auto worldSize = _simController->getWorldSize();
        ImGui::TextUnformatted(
            (StringHelper::format(worldSize.x) + " x " + StringHelper::format(worldSize.y)).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Zoom factor");
        ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(StringHelper::format(Viewport::getZoomFactor(), 2).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Center position");
        ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        auto centerPos = Viewport::getCenterInWorldPos();
        ImGui::TextUnformatted(
            (StringHelper::format(centerPos.x, 1) + ", " + StringHelper::format(centerPos.y, 1)).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        AlienImGui::Separator();
        AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Autotracking on selection"), _centerSelection);
        ImGui::Spacing();
        ImGui::Spacing();
        float sensitivity = Viewport::getZoomSensitivity();
        if (AlienImGui::SliderFloat(AlienImGui::SliderFloatParameters().name("Zoom sensitivity").min(1.0f).max(1.05f).textWidth(130).format(""), &sensitivity)) {
            Viewport::setZoomSensitivity(sensitivity);
        }
    }
    ImGui::EndChild();

    _resizeWorldDialog->process();
}

void _SpatialControlWindow::processBackground()
{
    processCenterOnSelection();
}

void _SpatialControlWindow::processZoomInButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_SEARCH_PLUS)) {
        Viewport::setZoomFactor(Viewport::getZoomFactor() * 2);
    }
    AlienImGui::Tooltip("Zoom in");
}

void _SpatialControlWindow::processZoomOutButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_SEARCH_MINUS)) {
        Viewport::setZoomFactor(Viewport::getZoomFactor() / 2);
    }
    AlienImGui::Tooltip("Zoom out");
}

void _SpatialControlWindow::processCenterButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_CROSSHAIRS)) {
        Viewport::setZoomFactor(1.0f);
        auto worldSize = toRealVector2D(_simController->getWorldSize());
        Viewport::setCenterInWorldPos({worldSize.x / 2, worldSize.y / 2});
    }
    AlienImGui::Tooltip("Center");
}

void _SpatialControlWindow::processResizeButton()
{
    if (AlienImGui::ToolbarButton(ICON_FA_CROP_ALT)) {
        _resizeWorldDialog->open();
    }
    AlienImGui::Tooltip("Resize");
}

void _SpatialControlWindow::processCenterOnSelection()
{
    if (_centerSelection && _simController->isSimulationRunning()) {
        auto shallowData = _simController->getSelectionShallowData();
        if (shallowData.numCells > 0 || shallowData.numParticles > 0) {
            Viewport::setCenterInWorldPos({shallowData.centerPosX, shallowData.centerPosY});
        }
    }
}
