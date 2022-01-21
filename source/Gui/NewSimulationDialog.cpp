#include "NewSimulationDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "Viewport.h"
#include "StatisticsWindow.h"
#include "AlienImGui.h"

_NewSimulationDialog::_NewSimulationDialog(
    SimulationController const& simController,
    Viewport const& viewport,
    StatisticsWindow const& statisticsWindow)
    : _simController(simController)
    , _viewport(viewport)
    , _statisticsWindow(statisticsWindow)
{}

void _NewSimulationDialog::process()
{
    if (!_on) {
        return;
    }
    ImGui::OpenPopup("New simulation");
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("New simulation", NULL, 0)) {
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

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            onNewSimulation();
            _on = false;
        }
        ImGui::SetItemDefaultFocus();

        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _on = false;
        }

        ImGui::EndPopup();
        _width = std::max(1, _width);
        _height = std::max(1, _height);
    }
}

void _NewSimulationDialog::show()
{
    _on = true;
    auto worldSize = _simController->getWorldSize();
    _width = worldSize.x;
    _height = worldSize.y;
}

void _NewSimulationDialog::onNewSimulation()
{
    _simController->closeSimulation();

    _statisticsWindow->reset();

    auto symbolMap = _simController->getSymbolMap();

    Settings settings;
    settings.generalSettings.worldSizeX = _width;
    settings.generalSettings.worldSizeY = _height;
    settings.simulationParameters = _simController->getSimulationParameters();
    settings.flowFieldSettings.centers[0].posX = toFloat(_width) / 2;
    settings.flowFieldSettings.centers[0].posY = toFloat(_height) / 2;

    _simController->newSimulation(0, settings, symbolMap);
    _viewport->setCenterInWorldPos({toFloat(_width) / 2, toFloat(_height) / 2});
    _viewport->setZoomFactor(4.0f);
}
