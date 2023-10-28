#include <imgui.h>

#include "ResizeWorldDialog.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"

_ResizeWorldDialog::_ResizeWorldDialog(SimulationController const& simController)
    : _AlienDialog("Resize world")
    , _simController(simController)
{}

void _ResizeWorldDialog::open()
{
    _AlienDialog::open();

    auto worldSize = _simController->getWorldSize();

    _width = worldSize.x;
    _height = worldSize.y;
}

void _ResizeWorldDialog::processIntern()
{
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_SizingStretchProp)) {

        //width
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputInt("##width", &_width);
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Width");

        //height
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
        ImGui::InputInt("##height", &_height);
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Height");

        ImGui::EndTable();
    }
    AlienImGui::ToggleButton(AlienImGui::ToggleButtonParameters().name("Scale content"), _scaleContent);

    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        onResizing();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }

    ImGui::EndPopup();
    _width = std::max(1, _width);
    _height = std::max(1, _height);
}

void _ResizeWorldDialog::onResizing()
{
    auto timestep = _simController->getCurrentTimestep();
    auto generalSettings = _simController->getGeneralSettings();
    auto parameters = _simController->getSimulationParameters();
    auto content = _simController->getClusteredSimulationData();

    _simController->closeSimulation();

    IntVector2D origWorldSize{generalSettings.worldSizeX, generalSettings.worldSizeY};
    generalSettings.worldSizeX = _width;
    generalSettings.worldSizeY = _height;

    _simController->newSimulation(timestep, generalSettings, parameters);

    DescriptionEditService::correctConnections(content, {_width, _height});
    if (_scaleContent) {
        DescriptionEditService::duplicate(content, origWorldSize, {_width, _height});
    }
    _simController->setClusteredSimulationData(content);
}
