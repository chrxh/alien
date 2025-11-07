#include "ResizeWorldDialog.h"

#include <imgui.h>

#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "TemporalControlWindow.h"
#include "Provider.h"

void ResizeWorldDialog::initIntern()
{

}

void ResizeWorldDialog::open()
{
    AlienDialog::open();

    auto worldSize = Provider::getSimulationFacade()->getWorldSize();

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
    auto timestep = Provider::getSimulationFacade()->getCurrentTimestep();
    auto worldSize = Provider::getSimulationFacade()->getWorldSize();
    auto parameters = Provider::getSimulationFacade()->getSimulationParameters();
    auto content = Provider::getSimulationFacade()->getSimulationData();
    auto realtime = Provider::getSimulationFacade()->getRealTime();
    auto const& statistics = Provider::getSimulationFacade()->getStatisticsHistory().getCopiedData();
    Provider::getSimulationFacade()->closeSimulation();

    IntVector2D origWorldSize = worldSize;
    worldSize.x = _width;
    worldSize.y = _height;

    Provider::getSimulationFacade()->newSimulation(timestep, worldSize, parameters);

    if (_scaleContent) {
        DescriptionEditService::get().duplicate(content, origWorldSize, {_width, _height});
    }
    Provider::getSimulationFacade()->setSimulationData(content);
    Provider::getSimulationFacade()->setStatisticsHistory(statistics);
    Provider::getSimulationFacade()->setRealTime(realtime);
    TemporalControlWindow::get().onSnapshot();
}
