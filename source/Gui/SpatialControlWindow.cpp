#include "SpatialControlWindow.h"

#include "imgui.h"

#include "Base/StringFormatter.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "OpenGLHelper.h"
#include "Resources.h"

_SpatialControlWindow::_SpatialControlWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    StyleRepository const& styleRepository)
    : _simController(simController)
    , _viewport(viewport)
    , _styleRepository(styleRepository)
{
    _zoomInTexture = OpenGLHelper::loadTexture(Const::ZoomInFilename);
    _zoomOutTexture = OpenGLHelper::loadTexture(Const::ZoomOutFilename);
}

void _SpatialControlWindow::process()
{
    if (!_on) {
        return;
    }

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Spatial control", &_on);

    processZoomInButton();
    ImGui::SameLine();
    processZoomOutButton();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::BeginTable(
            "##table1", 3, ImGuiTableFlags_SizingFixedFit /*ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg*/)) {

        //world size
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("World size");

        ImGui::TableSetColumnIndex(1);
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        auto worldSize = _simController->getWorldSize();
        ImGui::Text((StringFormatter::format(worldSize.x) + " x " + StringFormatter::format(worldSize.y)).c_str());
        ImGui::PopStyleColor();

        //zoom factor
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Zoom factor");

        ImGui::TableSetColumnIndex(1);
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::Text(StringFormatter::format(_viewport->getZoomFactor(), 1).c_str());
        ImGui::PopStyleColor();

        //view center position
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Center position");

        ImGui::TableSetColumnIndex(1);
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        auto centerPos = _viewport->getCenterInWorldPos();
        ImGui::Text(
            (StringFormatter::format(centerPos.x, 1) + ", " + StringFormatter::format(centerPos.y, 1)).c_str());
        ImGui::PopStyleColor();

        ImGui::EndTable();
    }

    processZoomSensitivitySlider();

    ImGui::End();
}

bool _SpatialControlWindow::isOn() const
{
    return _on;
}

void _SpatialControlWindow::setOn(bool value)
{
    _on = value;
}

void _SpatialControlWindow::processZoomInButton()
{
    if (ImGui::ImageButton((void*)(intptr_t)_zoomInTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _viewport->setZoomFactor(_viewport->getZoomFactor() * 2);
    }
}

void _SpatialControlWindow::processZoomOutButton()
{
    if (ImGui::ImageButton((void*)(intptr_t)_zoomOutTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _viewport->setZoomFactor(_viewport->getZoomFactor() / 2);
    }
}

void _SpatialControlWindow::processZoomSensitivitySlider()
{
    ImGui::Text("Zoom sensitivity");

    float sensitivity = _viewport->getZoomSensitivity();
    if (ImGui::SliderFloat("", &sensitivity, 1.0f, 1.15f, "")) {
        _viewport->setZoomSensitivity(sensitivity);
    }
}
