#include "CreatureTabWidget.h"
#include "CreatureTabWidget.h"

#include <imgui.h>

#include "Base/StringHelper.h"

#include "EngineInterface/GenomeDescriptionValidationService.h"

#include "AlienGui.h"
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

CreatureTabWidget _CreatureTabWidget::createPinnedCreatureTab(GenomeDescription_New const& genome, uint64_t creatureId)
{
    return CreatureTabWidget(new _CreatureTabWidget(genome, creatureId));
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

        AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->previewsHeight);

        if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
            processPreviews();
        }
        ImGui::EndChild();

        ImGui::PopID();
    }
    ImGui::EndChild();

    GenomeDescriptionValidationService::get().validateAndCorrect(_editData->genome);
}

void _CreatureTabWidget::onGenomeIntoCreatureInjected()
{
    _pinnedCreatureData->origGenome = _editData->genome;
}

bool _CreatureTabWidget::isDraft() const
{
    return !_pinnedCreatureData.has_value();
}

uint64_t _CreatureTabWidget::getCreatureId() const
{
    return _pinnedCreatureData->creatureId;
}

int _CreatureTabWidget::getTabId() const
{
    return _id;
}

std::string _CreatureTabWidget::getName() const
{
    if (isDraft()) {
        return "Draft " + std::to_string(_id);
    } else {
        auto result = "Creature " + StringHelper::formatInHex(_pinnedCreatureData->creatureId);
        if (_pinnedCreatureData->changesMade) {
            result = "* " + result;
        }
        return result;
    }
}

bool _CreatureTabWidget::hasCreaturesGenomeBeChanged() const
{
    if (!_pinnedCreatureData.has_value()) {
        return false;
    }
    return _pinnedCreatureData->changesMade;
}

GenomeDescription_New const& _CreatureTabWidget::getGenome()
{
    return _editData->genome;
}

bool _CreatureTabWidget::isEmpty() const
{
    return _editData->genome == GenomeDescription_New();
}

_CreatureTabWidget::_CreatureTabWidget(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData)
{
    static int _sequence = 0;
    _id = ++_sequence;

    _editData = std::make_shared<_CreatureTabEditData>(genome);
    _layoutData = layoutData;
    if (!_layoutData) {
        _layoutData = std::make_shared<_CreatureTabLayoutData>();
    }
    _genomeEditorWidget = _GenomeEditorWidget::create(_editData, _layoutData);
    _geneEditorWidget = _GeneEditorWidget::create(_editData, _layoutData);
    _nodeEditorWidget = _NodeEditorWidget::create(_editData, _layoutData);
}

_CreatureTabWidget::_CreatureTabWidget(GenomeDescription_New const& genome, uint64_t creatureId)
    : _CreatureTabWidget(genome, nullptr)
{
    _pinnedCreatureData = PinnedCreatureData{.creatureId = creatureId, .origGenome = genome};
}

void _CreatureTabWidget::processEditors()
{
    _genomeEditorWidget->process();

    ImGui::SameLine();
    ImGui::PushID(1);
    AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->genomeEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    _geneEditorWidget->process();

    ImGui::SameLine();
    ImGui::PushID(2);
    AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->geneEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    _nodeEditorWidget->process();

    if (_pinnedCreatureData.has_value()) {
        _pinnedCreatureData->changesMade = _pinnedCreatureData->origGenome != _editData->genome;
    }
}

void _CreatureTabWidget::processPreviews()
{
    if (ImGui::BeginChild("DesiredConfigurationPreview", ImVec2(_layoutData->desiredConfigurationPreviewWidth, 0))) {
        processDesiredConfigurationPreview();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    AlienGui::MovableVerticalSeparator(
        AlienGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->desiredConfigurationPreviewWidth);

    ImGui::SameLine();
    if (ImGui::BeginChild("ActualConfigurationPreview", ImVec2(0, 0))) {
        processActualConfigurationPreview();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processDesiredConfigurationPreview()
{
    AlienGui::Group("Preview (predicted)");
}

void _CreatureTabWidget::processActualConfigurationPreview()
{
    AlienGui::Group("Preview (simulated)");
}

void _CreatureTabWidget::doLayout()
{
    // Initial layout setup
    if (!_layoutData->initialized) {
        auto width = ImGui::GetContentRegionAvail().x;
        auto height = ImGui::GetContentRegionAvail().y;
        _layoutData->genomeEditorWidth = width / 3;
        _layoutData->geneEditorWidth = width / 3;
        _layoutData->previewsHeight = height / 2;
        _layoutData->desiredConfigurationPreviewWidth = width / 2;
        _layoutData->geneListHeight = height / 4;
        _layoutData->nodeListHeight = height / 4;
        _layoutData->neuralNetEditorHeight = height / 4;
        _layoutData->initialized = true;
        _origLayoutData = std::make_shared<_CreatureTabLayoutData>();
        *_origLayoutData = *_layoutData;
        return;
    }

    // Window size changes
    auto windowSize = ImGui::GetWindowSize();
    auto lastWindowSize = _lastWindowSize;
    _lastWindowSize = {windowSize.x, windowSize.y};
    if (lastWindowSize.has_value() && lastWindowSize->x > 0 && lastWindowSize->y > 0) {
        if (lastWindowSize->x != windowSize.x || lastWindowSize->y != windowSize.y) {
            auto scalingX = windowSize.x / lastWindowSize->x;
            auto scalingY = windowSize.y / lastWindowSize->y;
            _layoutData->genomeEditorWidth *= scalingX;
            _layoutData->geneEditorWidth *= scalingX;
            _layoutData->previewsHeight *= scalingY;
            _layoutData->desiredConfigurationPreviewWidth *= scalingX;
            _layoutData->geneListHeight *= scalingY;
            _layoutData->nodeListHeight *= scalingY;
            _layoutData->neuralNetEditorHeight *= scalingY;
            *_origLayoutData = *_layoutData;
            return;
        }
    }

    // Editor sizes changes
    if (_origLayoutData->genomeEditorWidth != _layoutData->genomeEditorWidth) {
        if (_layoutData->genomeEditorWidth < scale(200.0f) || _layoutData->geneEditorWidth < scale(200.0f)) {
            *_layoutData = *_origLayoutData;
            return;
        }
        _layoutData->geneEditorWidth += _origLayoutData->genomeEditorWidth - _layoutData->genomeEditorWidth;
    }
    if (_origLayoutData->geneEditorWidth != _layoutData->geneEditorWidth) {
        auto nodeEditorWidth = ImGui::GetContentRegionAvail().x - _layoutData->genomeEditorWidth - _layoutData->geneEditorWidth;
        if (_layoutData->geneEditorWidth < scale(200.0f) || nodeEditorWidth < scale(200.0f)) {
            *_layoutData = *_origLayoutData;
            return;
        }
    }
    if (_origLayoutData->previewsHeight != _layoutData->previewsHeight) {
        auto editorHeight = ImGui::GetContentRegionAvail().y - _layoutData->previewsHeight;
        if (_layoutData->previewsHeight < scale(200.0f) || editorHeight < scale(200.0f)) {
            *_layoutData = *_origLayoutData;
            return;
        }
    }

    *_origLayoutData = *_layoutData;
}
