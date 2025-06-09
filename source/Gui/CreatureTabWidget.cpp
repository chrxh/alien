#include "CreatureTabWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "CreatureTabEditData.h"
#include "CreatureTabLayoutData.h"
#include "GeneEditorWidget.h"
#include "GenomeEditorWidget.h"
#include "NodeEditorWidget.h"
#include "StyleRepository.h"

CreatureTabWidget _CreatureTabWidget::createDraftCreatureTab(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData)
{
    return CreatureTabWidget(new _CreatureTabWidget(genome, layoutData));
}

void _CreatureTabWidget::process()
{
    doLayout();

    if (ImGui::BeginChild("CreatureTab")) {
        ImGui::PushID(_id);

        if (ImGui::BeginChild("Editors", ImVec2(0, ImGui::GetContentRegionAvail().y - _layoutData->previewsHeight), 0)) {
            processEditors();
        }
        ImGui::EndChild();

        AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->previewsHeight);

        if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
            processPreviews();
        }
        ImGui::EndChild();

        ImGui::PopID();
    }
    ImGui::EndChild();
}

std::string _CreatureTabWidget::getName() const
{
    if (!_creatureId.has_value()) {
        return "Draft " + std::to_string(_id);
    }
    return "";
}

_CreatureTabWidget::_CreatureTabWidget(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData)
{
    static int _sequence = 0;
    _id = ++_sequence;

    _genomeData = std::make_shared<_CreatureTabEditData>(genome);
    _layoutData = layoutData;
    if (!_layoutData) {
        _layoutData = std::make_shared<_CreatureTabLayoutData>();
    }
    _genomeEditorWidget = _GenomeEditorWidget::create(_genomeData, _layoutData);
    _geneEditorWidget = _GeneEditorWidget::create(_genomeData, _layoutData);
    _nodeEditorWidget = _NodeEditorWidget::create(_genomeData, _layoutData);
}


void _CreatureTabWidget::processEditors()
{
    _genomeEditorWidget->process();

    ImGui::SameLine();
    ImGui::PushID(1);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->genomeEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    _geneEditorWidget->process();

    ImGui::SameLine();
    ImGui::PushID(2);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->geneEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    _nodeEditorWidget->process();
}

void _CreatureTabWidget::processPreviews()
{
    if (ImGui::BeginChild("DesiredConfigurationPreview", ImVec2(_layoutData->desiredConfigurationPreviewWidth, 0))) {
        processDesiredConfigurationPreview();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    AlienImGui::MovableVerticalSeparator(
        AlienImGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->desiredConfigurationPreviewWidth);

    ImGui::SameLine();
    if (ImGui::BeginChild("ActualConfigurationPreview", ImVec2(0, 0))) {
        processActualConfigurationPreview();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processDesiredConfigurationPreview()
{
    AlienImGui::Group("Preview (predicted)");
}

void _CreatureTabWidget::processActualConfigurationPreview()
{
    AlienImGui::Group("Preview (simulated)");
}

void _CreatureTabWidget::doLayout()
{
    if (_lastGenomeEditorWidth.has_value()) {
        _layoutData->geneEditorWidth += _lastGenomeEditorWidth.value() - _layoutData->genomeEditorWidth;
    }

    if (!_layoutData->initialized) {
        auto width = ImGui::GetContentRegionAvail().x;
        auto height = ImGui::GetContentRegionAvail().y;
        _layoutData->genomeEditorWidth = width / 3;
        _layoutData->geneEditorWidth = width / 3;
        _layoutData->previewsHeight = height / 2;
        _layoutData->desiredConfigurationPreviewWidth = width / 2;
        _layoutData->geneListHeight = height / 4;
        _layoutData->nodeListHeight = height / 4;
        _layoutData->initialized = true;
    }

    auto windowSize = ImGui::GetWindowSize();
    if (_lastWindowSize.has_value() && _lastWindowSize->x > 0 && _lastWindowSize->y > 0) {
        if (_lastWindowSize->x != windowSize.x || _lastWindowSize->y != windowSize.y) {
            auto scalingX = windowSize.x / _lastWindowSize->x;
            auto scalingY = windowSize.y / _lastWindowSize->y;
            _layoutData->genomeEditorWidth *= scalingX;
            _layoutData->geneEditorWidth *= scalingX;
            _layoutData->previewsHeight *= scalingY;
            _layoutData->desiredConfigurationPreviewWidth *= scalingX;
            _layoutData->geneListHeight *= scalingY;
            _layoutData->nodeListHeight *= scalingY;
        }
    }
    _lastWindowSize = {windowSize.x, windowSize.y};

    _layoutData->genomeEditorWidth = std::max(scale(50.0f), _layoutData->genomeEditorWidth);
    _layoutData->geneEditorWidth = std::max(scale(50.0f), _layoutData->geneEditorWidth);
    _layoutData->desiredConfigurationPreviewWidth = std::max(scale(50.0f), _layoutData->desiredConfigurationPreviewWidth);
    _layoutData->previewsHeight =
        std::min(ImGui::GetContentRegionAvail().y - scale(50.0f), std::max(scale(50.0f), _layoutData->previewsHeight));

    _lastGenomeEditorWidth = _layoutData->genomeEditorWidth;
}
