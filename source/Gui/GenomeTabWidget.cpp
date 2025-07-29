#include "GenomeTabWidget.h"
#include "GenomeTabWidget.h"

#include <imgui.h>

#include "Base/StringHelper.h"

#include "EngineInterface/GenomeDescriptionValidationService.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "GeneEditorWidget.h"
#include "GenomeEditorWidget.h"
#include "NodeEditorWidget.h"
#include "SimulatedPreviewWidget.h"
#include "StyleRepository.h"

GenomeTabWidget _GenomeTabWidget::createDraftTab(
    SimulationFacade const& simulationFacade,
    PreviewDescriptionSettings const& previewSettings,
    GenomeWindowEditData const& genomeEditData,
    GenomeDescription const& creature,
    GenomeTabLayoutData const& layoutData)
{
    return GenomeTabWidget(new _GenomeTabWidget(simulationFacade, previewSettings, genomeEditData, creature, DraftData(), layoutData));
}

GenomeTabWidget _GenomeTabWidget::createCreatureTab(
    SimulationFacade const& simulationFacade,
    PreviewDescriptionSettings const& previewSettings,
    GenomeWindowEditData const& genomeEditData,
    uint64_t creatureId,
    GenomeDescription const& genome,
    GenomeTabLayoutData const& layoutData)
{
    return GenomeTabWidget(new _GenomeTabWidget(
        simulationFacade, previewSettings, genomeEditData, genome, CreatureData{.creatureId = creatureId, .origGenome = genome}, layoutData));
}

void _GenomeTabWidget::process()
{
    doLayout();

    if (ImGui::BeginChild("CreatureTab")) {
        ImGui::PushID(_editData->id);

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

void _GenomeTabWidget::onGenomeIntoCreatureInjected()
{
    std::get<CreatureData>(_specificEditData).origGenome = _editData->genome;
}

bool _GenomeTabWidget::isDraft() const
{
    return std::holds_alternative<DraftData>(_specificEditData);
}

int _GenomeTabWidget::getTabId() const
{
    return _editData->id;
}

std::string _GenomeTabWidget::getName() const
{
    if (isDraft()) {
        return "Draft " + std::to_string(_editData->id);
    } else {
        auto const& simulatedCreatureData = std::get<CreatureData>(_specificEditData);
        auto result = "Creature " + StringHelper::formatInHex(simulatedCreatureData.creatureId);
        if (simulatedCreatureData.changesMade) {
            result = "* " + result;
        }
        return result;
    }
}

GenomeTabEditData const& _GenomeTabWidget::getEditData() const
{
    return _editData;
}

GenomeTabLayoutData const& _GenomeTabWidget::getLayoutData() const
{
    return _layoutData;
}

GenomeDescription const& _GenomeTabWidget::getGenomeDescription()
{
    return _editData->genome;
}


bool _GenomeTabWidget::hasCreaturesGenomeBeChanged() const
{
    if (!std::holds_alternative<CreatureData>(_specificEditData)) {
        return false;
    }
    auto const& simulatedCreatureData = std::get<CreatureData>(_specificEditData);
    return simulatedCreatureData.changesMade;
}

uint64_t _GenomeTabWidget::getCreatureId()
{
    auto const& simulatedCreatureData = std::get<CreatureData>(_specificEditData);
    return simulatedCreatureData.creatureId;
}

bool _GenomeTabWidget::isEmpty() const
{
    return _editData->genome == GenomeDescription();
}

void _GenomeTabWidget::convertToDraftTab()
{
    _specificEditData = DraftData{};
}

void _GenomeTabWidget::resetChanges()
{
    auto& simulatedCreatureData = std::get<CreatureData>(_specificEditData);
    simulatedCreatureData.origGenome = _editData->genome;
    simulatedCreatureData.changesMade = false;
}

_GenomeTabWidget::_GenomeTabWidget(
    SimulationFacade const& simulationFacade,
    PreviewDescriptionSettings const& previewSettings,
    GenomeWindowEditData const& genomeEditData,
    GenomeDescription const& genome,
    SpecificEditData const& specificEditData,
    GenomeTabLayoutData const& layoutData)
{
    static int _sequence = 0;

    _editData = std::make_shared<_GenomeTabEditData>(++_sequence, genome);
    if (!genome._genes.empty()) {
        _editData->selectedGeneIndex = 0;
        if (!genome._genes.front()._nodes.empty()) {
            _editData->selectedNodeByGeneIndex.emplace(0, 0);
        }
    }
    _layoutData = layoutData;
    if (!_layoutData) {
        _layoutData = std::make_shared<_GenomeTabLayoutData>();
    } else {
        _origLayoutData = _layoutData->clone();
    }
    _genomeEditorWidget = _GenomeEditorWidget::create(_editData, _layoutData);
    _geneEditorWidget = _GeneEditorWidget::create(_editData, _layoutData);
    _nodeEditorWidget = _NodeEditorWidget::create(_editData, _layoutData);
    _simulatedPreviewWidget = _SimulatedPreviewWidget::create(simulationFacade, previewSettings, genomeEditData, _editData);
    _specificEditData = specificEditData;
}

void _GenomeTabWidget::processEditors()
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

    if (std::holds_alternative<CreatureData>(_specificEditData)) {
        auto& simulatedCreatureData = std::get<CreatureData>(_specificEditData);
        simulatedCreatureData.changesMade = simulatedCreatureData.origGenome != _editData->genome;
    }
}

void _GenomeTabWidget::processPreviews()
{
    if (ImGui::BeginChild("DesiredConfigurationPreview", ImVec2(_layoutData->desiredConfigurationPreviewWidth, 0))) {
        processPredictedPreview();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    AlienGui::MovableVerticalSeparator(
        AlienGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->desiredConfigurationPreviewWidth);

    ImGui::SameLine();
    if (ImGui::BeginChild("ActualConfigurationPreview", ImVec2(0, 0))) {
        processSimulatedPreview();
    }
    ImGui::EndChild();
}

void _GenomeTabWidget::processPredictedPreview()
{
    AlienGui::Group("Preview (predicted)");
}

void _GenomeTabWidget::processSimulatedPreview()
{
    AlienGui::Group("Preview (simulated)");
    _simulatedPreviewWidget->process();
}

void _GenomeTabWidget::doLayout()
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
        _origLayoutData = _layoutData->clone();
       
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
