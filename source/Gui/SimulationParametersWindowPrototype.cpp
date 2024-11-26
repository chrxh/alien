#include "SimulationParametersWindowPrototype.h"

#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/SimulationFacade.h"

#include "AlienImGui.h"
#include "OverlayController.h"

SimulationParametersWindowPrototype::SimulationParametersWindowPrototype()
    : AlienWindow("Simulation parameters Prototype", "windows.simulation parameters prototype", false)
{}

void SimulationParametersWindowPrototype::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void SimulationParametersWindowPrototype::processIntern()
{
    processToolbar();
    processRegionMasterEditor();
    processRegionDetailEditor();
    processAddonList();
    processStatusBar();
}

void SimulationParametersWindowPrototype::processToolbar()
{
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
    }
    AlienImGui::Tooltip("Open simulation parameters from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
    }
    AlienImGui::Tooltip("Save simulation parameters to file");

    ImGui::SameLine();
    AlienImGui::ToolbarSeparator();

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
        _copiedParameters = _simulationFacade->getSimulationParameters();
        printOverlayMessage("Simulation parameters copied");
    }
    AlienImGui::Tooltip("Copy simulation parameters");

    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedParameters);
    if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
        _simulationFacade->setSimulationParameters(*_copiedParameters);
        _simulationFacade->setOriginalSimulationParameters(*_copiedParameters);
        printOverlayMessage("Simulation parameters pasted");
    }
    ImGui::EndDisabled();
    AlienImGui::Tooltip("Paste simulation parameters");

    AlienImGui::Separator();
}

void SimulationParametersWindowPrototype::processRegionMasterEditor()
{
}

void SimulationParametersWindowPrototype::processRegionDetailEditor()
{
}

void SimulationParametersWindowPrototype::processAddonList()
{
}

void SimulationParametersWindowPrototype::processStatusBar()
{
}
