#include "CreatureTabWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "CreatureTabLayoutData.h"

_CreatureTabWidget::_CreatureTabWidget(CreatureTabLayoutData const& creatureTabLayoutData)
    : _creatureTabLayoutData(creatureTabLayoutData)
{
}

void _CreatureTabWidget::process()
{
    if (ImGui::BeginChild("Editors", ImVec2(0, ImGui::GetContentRegionAvail().y - _creatureTabLayoutData->_previewsHeight), 0)) {
        processEditors();
    }
    ImGui::EndChild();

    AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _creatureTabLayoutData->_previewsHeight);

    if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        processPreviews();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processEditors()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_creatureTabLayoutData->_genomeEditorWidth, 0))) {
        processGenomeEditor();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(1);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _creatureTabLayoutData->_genomeEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    if (ImGui::BeginChild("GeneEditor", ImVec2(_creatureTabLayoutData->_geneEditorWidth, 0))) {
        processGeneEditor();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(2);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _creatureTabLayoutData->_geneEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        processNodeEditor();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processPreviews()
{
    if (ImGui::BeginChild("DesiredConfigurationPreview", ImVec2(_creatureTabLayoutData->_desiredConfigurationPreviewWidth, 0))) {
        processDesiredConfigurationPreview();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    AlienImGui::MovableVerticalSeparator(
        AlienImGui::MovableVerticalSeparatorParameters().additive(true), _creatureTabLayoutData->_desiredConfigurationPreviewWidth);

    ImGui::SameLine();
    if (ImGui::BeginChild("ActualConfigurationPreview", ImVec2(0, 0))) {
        processActualConfigurationPreview();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processGenomeEditor()
{
    AlienImGui::Group("Genome");
}

void _CreatureTabWidget::processGeneEditor()
{
    AlienImGui::Group("Selected gene");
}

void _CreatureTabWidget::processNodeEditor()
{
    AlienImGui::Group("Selected node");
}

void _CreatureTabWidget::processDesiredConfigurationPreview()
{
    AlienImGui::Group("Preview (desired)");
}

void _CreatureTabWidget::processActualConfigurationPreview()
{
    AlienImGui::Group("Preview (actual)");
}
