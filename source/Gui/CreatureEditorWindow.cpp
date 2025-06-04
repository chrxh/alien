#include "CreatureEditorWindow.h"

#include <Fonts/IconsFontAwesome5.h>

#include "AlienImGui.h"
 
CreatureEditorWindow::CreatureEditorWindow()
    : AlienWindow("Creature editor", "windows.creature editor", false, true)
{
}

void CreatureEditorWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void CreatureEditorWindow::shutdownIntern()
{
}

void CreatureEditorWindow::processIntern()
{
    processToolbar();
    processGenomeEditor();
    processPreviews();
}

void CreatureEditorWindow::processToolbar()
{
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_FOLDER_OPEN))) {
    }
    AlienImGui::Tooltip("Open creature from file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_SAVE))) {
    }
    AlienImGui::Tooltip("Save creature to file");

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(AlienImGui::ToolbarButtonParameters().text(ICON_FA_UPLOAD))) {
    }
    AlienImGui::Tooltip("Share your creature with other users:\nYour current creature will be uploaded to the server and made visible in the browser.");

    AlienImGui::Separator();
}

void CreatureEditorWindow::processGenomeEditor()
{
}

void CreatureEditorWindow::processPreviews()
{
}
