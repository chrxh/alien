#include "GenomeTabWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/DescValidationService.h>
#include <EngineInterface/GenomeDescInfoService.h>

#include "AlienGui.h"
#include "GeneEditorWidget.h"
#include "GenomeEditorWidget.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "GenomeTabWidget.h"
#include "NodeEditorWidget.h"
#include "PreviewWidget.h"
#include "StyleRepository.h"

GenomeTabWidget _GenomeTabWidget::create(
    GenomeWindowEditData const& genomeEditData,
    GenomeDesc const& genome,
    GenomeTabLayoutData const& layoutData,
    std::optional<int> lineageId)
{
    return GenomeTabWidget(new _GenomeTabWidget(genomeEditData, genome, layoutData, lineageId));
}

GenomeDesc _GenomeTabWidget::normalizeForEditor(GenomeDesc genome)
{
    DescValidationService::get().validateAndCorrect(genome);
    for (auto& gene : genome._genes) {
        _GenomeTabEditData::updateGeneGeometry(gene);
    }
    return genome;
}

void _GenomeTabWidget::process()
{
    doLayout();

    if (ImGui::BeginChild("CreatureTab")) {
        ImGui::PushID(_editData->id);

        processBreadcrumb();

        auto statusBarHeight = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
        if (ImGui::BeginChild("Editors", ImVec2(0, -statusBarHeight), 0)) {
            processEditors();
            DescValidationService::get().validateAndCorrect(_editData->genome);
        }
        ImGui::EndChild();

        processStatusBar();

        ImGui::PopID();
    }
    ImGui::EndChild();
}

int _GenomeTabWidget::getTabId() const
{
    return _editData->id;
}

std::string _GenomeTabWidget::getName() const
{
    std::string result;
    if (_editData->changesMade) {
        result = "* ";
    }
    result += _editData->genome._name;
    return result;
}

std::optional<int> _GenomeTabWidget::getLineageId() const
{
    return _lineageId;
}

void _GenomeTabWidget::setLineageId(std::optional<int> value)
{
    _lineageId = value;
}

GenomeTabEditData const& _GenomeTabWidget::getEditData() const
{
    return _editData;
}

GenomeTabLayoutData const& _GenomeTabWidget::getLayoutData() const
{
    return _layoutData;
}

GenomeDesc const& _GenomeTabWidget::getGenomeDesc() const
{
    return _editData->genome;
}

void _GenomeTabWidget::setGenomeDesc(GenomeDesc const& genome)
{
    _editData->genome = genome;
}

bool _GenomeTabWidget::hasGenomeChanged() const
{
    return _editData->changesMade;
}

bool _GenomeTabWidget::isEmpty() const
{
    return _editData->genome == GenomeDesc();
}

void _GenomeTabWidget::resetOriginal()
{
    _editData->origGenome = _editData->genome;
    _editData->changesMade = false;
}

void _GenomeTabWidget::revertChanges()
{
    _editData->genome = _editData->origGenome;
    _editData->changesMade = false;

    // Maintain selection only for valid nodes and genes
    if (_editData->selectedGeneIndex) {
        if (_editData->genome._genes.size() <= _editData->selectedGeneIndex.value()) {
            _editData->selectedGeneIndex.reset();
        }
    }
    std::map<int, int> newSelectedNodeByGeneIndex;
    for (auto& [geneIndex, nodeIndex] : _editData->selectedNodeByGeneIndex) {
        if (_editData->genome._genes.size() > geneIndex) {
            if (_editData->genome._genes.at(geneIndex)._nodes.size() > nodeIndex) {
                newSelectedNodeByGeneIndex.emplace(geneIndex, nodeIndex);
            }
        }
    }
    _editData->selectedNodeByGeneIndex = newSelectedNodeByGeneIndex;
}

_GenomeTabWidget::_GenomeTabWidget(
    GenomeWindowEditData const& genomeEditData,
    GenomeDesc const& genome,
    GenomeTabLayoutData const& layoutData,
    std::optional<int> lineageId)
    : _lineageId(lineageId)
{
    static int _sequence = 0;

    _editData = std::make_shared<_GenomeTabEditData>(++_sequence, genome);
    _editData->id = ++_sequence;

    auto normalizedGenome = normalizeForEditor(genome);

    _editData->genome = normalizedGenome;
    _editData->origGenome = normalizedGenome;

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
    _simulatedPreviewWidget = _PreviewWidget::create(genomeEditData, _editData);
}

void _GenomeTabWidget::processBreadcrumb()
{
    std::vector<std::string> items;
    items.emplace_back(_editData->genome._name.empty() ? "(unnamed genome)" : _editData->genome._name);

    if (_editData->selectedGeneIndex.has_value() && _editData->hasValidGeneIndex(_editData->selectedGeneIndex.value())) {
        auto const& gene = _editData->genome._genes.at(_editData->selectedGeneIndex.value());
        items.emplace_back(gene._name.empty() ? "Gene " + std::to_string(_editData->selectedGeneIndex.value()) : gene._name);

        if (_editData->isNodeLevelSelected()) {
            items.emplace_back("Node " + std::to_string(_editData->getSelectedNodeIndex().value()));
        }
    }

    AlienGui::Breadcrumb(items);
    AlienGui::Separator();
}

void _GenomeTabWidget::processEditors()
{
    // Left field: genome properties, mutation rates and the gene/node tree
    _genomeEditorWidget->process();

    ImGui::SameLine();
    ImGui::PushID(1);
    AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->genomeEditorWidth);
    ImGui::PopID();

    // Middle field: the inspector follows the selection and shows either the gene or the node level
    ImGui::SameLine();
    if (ImGui::BeginChild("Inspector", ImVec2(_layoutData->inspectorWidth, 0))) {
        if (_editData->isNodeLevelSelected()) {
            _nodeEditorWidget->process();
        } else {
            _geneEditorWidget->process();
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(2);
    AlienGui::MovableVerticalSeparator(AlienGui::MovableVerticalSeparatorParameters().additive(true), _layoutData->inspectorWidth);
    ImGui::PopID();

    // Right field: the creature previews
    ImGui::SameLine();
    if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        processPreview();
    }
    ImGui::EndChild();

    _editData->changesMade = !_editData->origGenome.equalWithoutId(_editData->genome);
}

void _GenomeTabWidget::processPreview()
{
    _simulatedPreviewWidget->process();
}

void _GenomeTabWidget::processStatusBar()
{
    auto const& genome = _editData->genome;
    auto numGenes = genome._genes.size();
    auto numNodes = GenomeDescInfoService::get().getNumberOfNodes(genome);

    std::string text =
        std::to_string(numGenes) + (numGenes == 1 ? " gene" : " genes") + "  \xC2\xB7  " + std::to_string(numNodes) + (numNodes == 1 ? " node" : " nodes");
    if (_editData->selectedGeneIndex.has_value()) {
        text += "  \xC2\xB7  gene " + std::to_string(_editData->selectedGeneIndex.value());
        if (_editData->isNodeLevelSelected()) {
            text += ", node " + std::to_string(_editData->getSelectedNodeIndex().value());
        }
    }
    AlienGui::Text(AlienGui::TextParameters().text(text).style(AlienGui::TextStyle::Decent));
}

void _GenomeTabWidget::doLayout()
{
    auto minColumnWidth = scale(240.0f);
    auto minSectionHeight = scale(120.0f);

    // Initial layout setup
    if (!_layoutData->initialized) {
        auto width = ImGui::GetContentRegionAvail().x;
        auto height = ImGui::GetContentRegionAvail().y;
        _layoutData->genomeEditorWidth = width / 4;
        _layoutData->inspectorWidth = width / 4;
        _layoutData->desiredConfigurationPreviewWidth = width / 2;
        _layoutData->structureHeight = height / 2;
        _layoutData->neuralNetEditorHeight = height / 3;
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
            _layoutData->inspectorWidth *= scalingX;
            _layoutData->desiredConfigurationPreviewWidth *= scalingX;
            _layoutData->structureHeight *= scalingY;
            _layoutData->neuralNetEditorHeight *= scalingY;
            *_origLayoutData = *_layoutData;
            return;
        }
    }

    // Editor size changes
    auto previewWidth = ImGui::GetContentRegionAvail().x - _layoutData->genomeEditorWidth - _layoutData->inspectorWidth;
    if (_origLayoutData->genomeEditorWidth != _layoutData->genomeEditorWidth || _origLayoutData->inspectorWidth != _layoutData->inspectorWidth) {
        if (_layoutData->genomeEditorWidth < minColumnWidth || _layoutData->inspectorWidth < minColumnWidth || previewWidth < minColumnWidth) {
            *_layoutData = *_origLayoutData;
            return;
        }
    }
    if (_origLayoutData->structureHeight != _layoutData->structureHeight) {
        auto headerHeight = ImGui::GetContentRegionAvail().y - _layoutData->structureHeight;
        if (_layoutData->structureHeight < minSectionHeight || headerHeight < minSectionHeight) {
            *_layoutData = *_origLayoutData;
            return;
        }
    }

    *_origLayoutData = *_layoutData;
}
