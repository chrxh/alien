#include "ResizeWorldDialog.h"

#include <imgui.h>

#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "TemporalControlWindow.h"
#include <EngineInterface/SimulationFacade.h>

void ResizeWorldDialog::initIntern()
{

}

void ResizeWorldDialog::open()
{
    AlienDialog::open();

    auto worldSize = _SimulationFacade::get()->getWorldSize();

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
    auto timestep = _SimulationFacade::get()->getCurrentTimestep();
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto content = _SimulationFacade::get()->getSimulationData();
    auto realtime = _SimulationFacade::get()->getRealTime();
    auto const& statistics = _SimulationFacade::get()->getStatisticsHistory().getCopiedData();
    _SimulationFacade::get()->closeSimulation();

    IntVector2D origWorldSize = worldSize;
    worldSize.x = _width;
    worldSize.y = _height;

    _SimulationFacade::get()->newSimulation(timestep, worldSize, parameters);

    if (_scaleContent) {
        DescriptionEditService::get().duplicate(content, origWorldSize, {_width, _height});
    }
    _SimulationFacade::get()->setSimulationData(content);
    _SimulationFacade::get()->setStatisticsHistory(statistics);
    _SimulationFacade::get()->setRealTime(realtime);
    TemporalControlWindow::get().onSnapshot();
}
