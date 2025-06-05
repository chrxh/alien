#include "CreatureTabWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "CreatureTabLayoutData.h"
#include "StyleRepository.h"

_CreatureTabWidget::_CreatureTabWidget(GenomeDescription_New const& genome, std::optional<CreatureTabLayoutData> const& creatureTabLayoutData)
    : _genome(genome)
{
    static int _sequence = 0;
    _id = ++_sequence;

    _creatureTabLayoutData = creatureTabLayoutData;
}

void _CreatureTabWidget::process()
{
    correctingLayout();

    ImGui::PushID(_id);

    if (ImGui::BeginChild("Editors", ImVec2(0, ImGui::GetContentRegionAvail().y - _creatureTabLayoutData->_previewsHeight), 0)) {
        processEditors();
    }
    ImGui::EndChild();

    AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _creatureTabLayoutData->_previewsHeight);

    if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        processPreviews();
    }
    ImGui::EndChild();

    ImGui::PopID();
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

void _CreatureTabWidget::correctingLayout()
{
    if (_lastGenomeEditorWidth.has_value()) {
        _creatureTabLayoutData->_geneEditorWidth += _lastGenomeEditorWidth.value() - _creatureTabLayoutData->_genomeEditorWidth;
    }

    if (!_creatureTabLayoutData.has_value()) {
        auto width = ImGui::GetContentRegionAvail().x;
        auto height = ImGui::GetContentRegionAvail().y;
        CreatureTabLayoutData creatureTabLayoutData;
        creatureTabLayoutData._genomeEditorWidth = width / 3;
        creatureTabLayoutData._geneEditorWidth = width / 3;
        creatureTabLayoutData._previewsHeight = height / 2;
        creatureTabLayoutData._desiredConfigurationPreviewWidth = width / 2;
        _creatureTabLayoutData = creatureTabLayoutData;
    }

    auto windowSize = ImGui::GetWindowSize();
    if (_lastWindowSize.has_value() && _lastWindowSize->x > 0 && _lastWindowSize->y > 0) {
        if (_lastWindowSize->x != windowSize.x || _lastWindowSize->y != windowSize.y) {
            auto scalingX = windowSize.x / _lastWindowSize->x;
            auto scalingY = windowSize.y / _lastWindowSize->y;
            _creatureTabLayoutData->_genomeEditorWidth *= scalingX;
            _creatureTabLayoutData->_geneEditorWidth *= scalingX;
            _creatureTabLayoutData->_desiredConfigurationPreviewWidth *= scalingX;
            _creatureTabLayoutData->_previewsHeight *= scalingY;
        }
    }
    _lastWindowSize = {windowSize.x, windowSize.y};

    _creatureTabLayoutData->_genomeEditorWidth = std::max(scale(50.0f), _creatureTabLayoutData->_genomeEditorWidth);
    _creatureTabLayoutData->_geneEditorWidth = std::max(scale(50.0f), _creatureTabLayoutData->_geneEditorWidth);
    _creatureTabLayoutData->_desiredConfigurationPreviewWidth = std::max(scale(50.0f), _creatureTabLayoutData->_desiredConfigurationPreviewWidth);
    _creatureTabLayoutData->_previewsHeight =
        std::min(ImGui::GetContentRegionAvail().y - scale(50.0f), std::max(scale(50.0f), _creatureTabLayoutData->_previewsHeight));

    _lastGenomeEditorWidth = _creatureTabLayoutData->_genomeEditorWidth;
}
