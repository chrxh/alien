#include <imgui.h>

#include "ResizeWorldDialog.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienImGui.h"
#include "TemporalControlWindow.h"

_ResizeWorldDialog::_ResizeWorldDialog(SimulationFacade const& simulationFacade, TemporalControlWindow const& temporalControlWindow)
    : AlienDialog("Resize world")
    , _simulationFacade(simulationFacade)
    , _temporalControlWindow(temporalControlWindow)
{}

void _ResizeWorldDialog::open()
{
    AlienDialog::open();

    auto worldSize = _simulationFacade->getWorldSize();

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

    _width = std::max(1, _width);
    _height = std::max(1, _height);
}

void _ResizeWorldDialog::onResizing()
{
    auto name = _simulationFacade->getSimulationName();
    auto timestep = _simulationFacade->getCurrentTimestep();
    auto generalSettings = _simulationFacade->getGeneralSettings();
    auto parameters = _simulationFacade->getSimulationParameters();
    auto content = _simulationFacade->getClusteredSimulationData();
    auto realtime = _simulationFacade->getRealTime();
    auto const& statistics = _simulationFacade->getStatisticsHistory().getCopiedData();
    _simulationFacade->closeSimulation();

    IntVector2D origWorldSize{generalSettings.worldSizeX, generalSettings.worldSizeY};
    generalSettings.worldSizeX = _width;
    generalSettings.worldSizeY = _height;

    _simulationFacade->newSimulation(name, timestep, generalSettings, parameters);

    DescriptionEditService::correctConnections(content, {_width, _height});
    if (_scaleContent) {
        DescriptionEditService::duplicate(content, origWorldSize, {_width, _height});
    }
    _simulationFacade->setClusteredSimulationData(content);
    _simulationFacade->setStatisticsHistory(statistics);
    _simulationFacade->setRealTime(realtime);
    _temporalControlWindow->onSnapshot();
}
