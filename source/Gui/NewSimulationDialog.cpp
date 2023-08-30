#include "NewSimulationDialog.h"

#include <imgui.h>

#include "EngineInterface/SimulationController.h"
#include "Viewport.h"
#include "StatisticsWindow.h"
#include "TemporalControlWindow.h"
#include "AlienImGui.h"
#include "GlobalSettings.h"

namespace
{
    auto const ContentTextInputWidth = 60.0f;
}

_NewSimulationDialog::_NewSimulationDialog(
    SimulationController const& simController,
    TemporalControlWindow const& temporalControlWindow,
    Viewport const& viewport,
    StatisticsWindow const& statisticsWindow)
    : _AlienDialog("New simulation")
    , _simController(simController)
    , _temporalControlWindow(temporalControlWindow)
    , _viewport(viewport)
    , _statisticsWindow(statisticsWindow)
{
    _adoptSimulationParameters = GlobalSettings::getInstance().getBoolState("dialogs.new simulation.adopt simulation parameters", true);
}

_NewSimulationDialog::~_NewSimulationDialog()
{
    GlobalSettings::getInstance().setBoolState("dialogs.new simulation.adopt simulation parameters", _adoptSimulationParameters);
}

void _NewSimulationDialog::processIntern()
{
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Width").textWidth(ContentTextInputWidth), _width);
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Height").textWidth(ContentTextInputWidth), _height);
    AlienImGui::Checkbox(
        AlienImGui::CheckboxParameters().name("Adopt simulation parameters").textWidth(0), _adoptSimulationParameters);

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
    auto worldSize = _simController->getWorldSize();
    _width = worldSize.x;
    _height = worldSize.y;
}

void _NewSimulationDialog::onNewSimulation()
{
    SimulationParameters parameters;
    if (_adoptSimulationParameters) {
        parameters = _simController->getSimulationParameters();
    }
    _simController->closeSimulation();

    _statisticsWindow->reset();

    GeneralSettings generalSettings;
    generalSettings.worldSizeX = _width;
    generalSettings.worldSizeY = _height;
    _simController->newSimulation(0, generalSettings, parameters);
    _viewport->setCenterInWorldPos({toFloat(_width) / 2, toFloat(_height) / 2});
    _viewport->setZoomFactor(4.0f);
    _temporalControlWindow->onSnapshot();
}
