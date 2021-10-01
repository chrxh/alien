#include "SpatialControlWindow.h"

#include "imgui.h"

#include "Base/StringFormatter.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/DescriptionHelper.h"
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
    _resizeTexture = OpenGLHelper::loadTexture(Const::ZoomOutFilename);
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
    ImGui::SameLine();
    processResizeButton();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::BeginTable(
            "##table1", 2, ImGuiTableFlags_SizingStretchProp /*ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg*/)) {

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

    processResizeDialog();
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

void _SpatialControlWindow::processResizeButton()
{
    if (ImGui::ImageButton((void*)(intptr_t)_resizeTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _showResizeDialog = true;
        auto worldSize = _simController->getWorldSize();
        _width = worldSize.x;
        _height = worldSize.y;
    }
}

void _SpatialControlWindow::processZoomSensitivitySlider()
{
    ImGui::Text("Zoom sensitivity");

    float sensitivity = _viewport->getZoomSensitivity();
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::SliderFloat("", &sensitivity, 1.0f, 1.15f, "")) {
        _viewport->setZoomSensitivity(sensitivity);
    }
    ImGui::PopItemWidth();
}

void _SpatialControlWindow::processResizeDialog()
{
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();

    if (_showResizeDialog) {
        ImGui::OpenPopup("Resize world");
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("Resize world", NULL, 0)) {
            if (ImGui::BeginTable(
                    "##", 2, ImGuiTableFlags_SizingStretchProp)) {

                //width
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Width");

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
                ImGui::InputInt("##width", &_width);
                ImGui::PopItemWidth();

                //height
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Height");

                ImGui::TableSetColumnIndex(1);
                ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
                ImGui::InputInt("##height", &_height);
                ImGui::PopItemWidth();

                ImGui::EndTable();
            }
            ImGui::Checkbox("Scale content", &_scaleContent);

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::Spacing();

            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
                _showResizeDialog = false;
                onResizing();
            }
            ImGui::SetItemDefaultFocus();

            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
                _showResizeDialog = false;
            }

            ImGui::EndPopup();
        }
    }
}

void _SpatialControlWindow::onResizing()
{
    auto timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
    auto generalSettings = _simController->getGeneralSettings();
    auto simulationParameters = _simController->getSimulationParameters();
    auto symbolMap = _simController->getSymbolMap();
    auto content = _simController->getSimulationData({0, 0}, _simController->getWorldSize());

    _simController->closeSimulation();

    auto origWorldSize = generalSettings.worldSize;
    generalSettings.worldSize = {_width, _height};
    _simController->newSimulation(
        timestep,
        generalSettings,
        simulationParameters,
        symbolMap);

    DescriptionHelper::correctConnections(content, generalSettings.worldSize);
    if (_scaleContent) {
        DescriptionHelper::duplicate(content, origWorldSize, generalSettings.worldSize);
    }
    _simController->updateData(content);
}
