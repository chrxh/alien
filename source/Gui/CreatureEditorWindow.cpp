#include "CreatureEditorWindow.h"

#include <Fonts/IconsFontAwesome5.h>

#include "AlienImGui.h"
#include "EditorController.h"

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

    _previewsHeight = std::min(ImGui::GetContentRegionAvail().y - scale(10.0f), std::max(scale(10.0f), _previewsHeight));

    if (ImGui::BeginChild("Editors", ImVec2(0, ImGui::GetContentRegionAvail().y - _previewsHeight), 0)) {
        processEditors();
    }
    ImGui::EndChild();

    AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _previewsHeight);

    if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        processPreviews();
    }
    ImGui::EndChild();
}

bool CreatureEditorWindow::isShown()
{
    return _on && EditorController::get().isOn();
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

void CreatureEditorWindow::processEditors()
{
    _genomeEditorWidth = std::max(scale(10.0f), _genomeEditorWidth);
    _geneEditorWidth = std::max(scale(10.0f), _geneEditorWidth);
    auto origGenomeEditorWidth = _genomeEditorWidth;

    if (ImGui::BeginChild("GenomeEditor", ImVec2(_genomeEditorWidth, 0), ImGuiChildFlags_Border)) {
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(1);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _genomeEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    if (ImGui::BeginChild("GeneEditor", ImVec2(_geneEditorWidth, 0), ImGuiChildFlags_Border)) {
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(2);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _geneEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0), ImGuiChildFlags_Border)) {
    }
    ImGui::EndChild();
    _geneEditorWidth += origGenomeEditorWidth - _genomeEditorWidth;
}

void CreatureEditorWindow::processPreviews()
{
}
