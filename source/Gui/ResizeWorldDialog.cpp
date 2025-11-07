#include "ResizeWorldDialog.h"

#include <imgui.h>

#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "TemporalControlWindow.h"
#include "SimulationFacadeProvider.h"

void ResizeWorldDialog::initIntern()
{

}

void ResizeWorldDialog::open()
{
    AlienDialog::open();

    auto worldSize = SimulationFacadeProvider::getSimulationFacade()->getWorldSize();

    _width = worldSize.x;
    _height = worldSize.y;
}

ResizeWorldDialog::ResizeWorldDialog()
    : AlienDialog("Resize world")
{}

void ResizeWorldDialog::processIntern()
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
    AlienGui::ToggleButton(AlienGui::ToggleButtonParameters().name("Scale content"), _scaleContent);

    AlienGui::Separator();

    if (AlienGui::Button("OK")) {
        onResizing();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }

    _width = std::max(1, _width);
    _height = std::max(1, _height);
}

void ResizeWorldDialog::onResizing()
{
    auto timestep = SimulationFacadeProvider::getSimulationFacade()->getCurrentTimestep();
    auto worldSize = SimulationFacadeProvider::getSimulationFacade()->getWorldSize();
    auto parameters = SimulationFacadeProvider::getSimulationFacade()->getSimulationParameters();
    auto content = SimulationFacadeProvider::getSimulationFacade()->getSimulationData();
    auto realtime = SimulationFacadeProvider::getSimulationFacade()->getRealTime();
    auto const& statistics = SimulationFacadeProvider::getSimulationFacade()->getStatisticsHistory().getCopiedData();
    SimulationFacadeProvider::getSimulationFacade()->closeSimulation();

    IntVector2D origWorldSize = worldSize;
    worldSize.x = _width;
    worldSize.y = _height;

    SimulationFacadeProvider::getSimulationFacade()->newSimulation(timestep, worldSize, parameters);

    if (_scaleContent) {
        DescriptionEditService::get().duplicate(content, origWorldSize, {_width, _height});
    }
    SimulationFacadeProvider::getSimulationFacade()->setSimulationData(content);
    SimulationFacadeProvider::getSimulationFacade()->setStatisticsHistory(statistics);
    SimulationFacadeProvider::getSimulationFacade()->setRealTime(realtime);
    TemporalControlWindow::get().onSnapshot();
}
