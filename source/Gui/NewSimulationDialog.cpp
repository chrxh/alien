#include "NewSimulationDialog.h"

#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "EngineInterface/SimulationFacade.h"

#include "Viewport.h"
#include "StatisticsWindow.h"
#include "TemporalControlWindow.h"
#include "AlienImGui.h"
#include "StyleRepository.h"

namespace
{
    auto const ContentTextInputWidth = 60.0f;
}

_NewSimulationDialog::_NewSimulationDialog(
    SimulationFacade const& simulationFacade,
    TemporalControlWindow const& temporalControlWindow,
    StatisticsWindow const& statisticsWindow)
    : AlienDialog("New simulation")
    , _simulationFacade(simulationFacade)
    , _temporalControlWindow(temporalControlWindow)
    , _statisticsWindow(statisticsWindow)
{
    _adoptSimulationParameters = GlobalSettings::get().getBool("dialogs.new simulation.adopt simulation parameters", true);
}

_NewSimulationDialog::~_NewSimulationDialog()
{
    GlobalSettings::get().setBool("dialogs.new simulation.adopt simulation parameters", _adoptSimulationParameters);
}

void _NewSimulationDialog::processIntern()
{
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Width").textWidth(ContentTextInputWidth), _width);
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Height").textWidth(ContentTextInputWidth), _height);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters().name("Adopt simulation parameters").textWidth(0), _adoptSimulationParameters);

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();
    if (AlienImGui::Button("OK")) {
        ImGui::CloseCurrentPopup();
        onNewSimulation();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
        close();
    }

    _width = std::max(1, _width);
    _height = std::max(1, _height);
}

void _NewSimulationDialog::openIntern()
{
    auto worldSize = _simulationFacade->getWorldSize();
    _width = worldSize.x;
    _height = worldSize.y;
}

void _NewSimulationDialog::onNewSimulation()
{
    SimulationParameters parameters;
    if (_adoptSimulationParameters) {
        parameters = _simulationFacade->getSimulationParameters();
    }
    _simulationFacade->closeSimulation();

    GeneralSettings generalSettings;
    generalSettings.worldSizeX = _width;
    generalSettings.worldSizeY = _height;
    _simulationFacade->newSimulation(std::nullopt, 0, generalSettings, parameters);
    Viewport::get().setCenterInWorldPos({toFloat(_width) / 2, toFloat(_height) / 2});
    Viewport::get().setZoomFactor(4.0f);
    _temporalControlWindow->onSnapshot();
}
