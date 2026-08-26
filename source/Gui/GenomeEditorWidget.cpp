#include "GenomeEditorWidget.h"

#include <ranges>
#include <set>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/GenomeDescEditService.h>
#include <EngineInterface/GenomeDescInfoService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "GenericMessageDialog.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "MutationRatesWidget.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;

    auto constexpr ColorChipSize = 9.0f;
    auto constexpr ColorChipSpacing = 3.0f;
}


GenomeEditorWidget _GenomeEditorWidget::create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
{
    return GenomeEditorWidget(new _GenomeEditorWidget(editData, layoutData));
}

void _GenomeEditorWidget::process()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_layoutData->genomeEditorWidth, 0))) {
        processHeaderData();

        AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->structureHeight);

        processStructureTree();
        processStructureButtons();
    }
    ImGui::EndChild();
}

_GenomeEditorWidget::_GenomeEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
    : _editData(editData)
    , _layoutData(layoutData)
{
    for (auto const& gene : editData->genome._genes) {
        try {
            auto name = gene._name;
            std::string const prefix = "Gene ";
            if (name.starts_with(prefix)) {
                std::string numberPart = name.substr(prefix.size());
                int number = std::stoi(numberPart);
                _sequenceNumberForCreatedGenes = std::max(_sequenceNumberForCreatedGenes, number + 1);
            }
        } catch (...) {
        }
    }
}

void _GenomeEditorWidget::processHeaderData()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Genome").highlighted(true));

    if (ImGui::BeginChild("GenomeHeader", ImVec2(0, -_layoutData->structureHeight), 0)) {

        auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

        AlienGui::InputText(AlienGui::InputTextParameters().name("Genome name").textWidth(rightColumnWidth), _editData->genome._name);

        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Front angle").format("%.1f").min(-180.0f).max(180.0f).textWidth(rightColumnWidth),
            &_editData->genome._frontAngle);

        AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Resistance to injection").textWidth(rightColumnWidth), _editData->genome._resistanceToInjection);

        AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Apply meta-mutations").textWidth(rightColumnWidth), _editData->genome._applyMetaMutations);

        AlienGui::Group(AlienGui::GroupParameters().text("Mutation rates"));

        _mutationRatesWidget.process(_editData->genome._mutationRates, rightColumnWidth);
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::processStructureTree()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Structure"));

    if (ImGui::BeginChild("Structure", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()))) {
        auto scrollToSelection = _selectedGeneFromPreviousFrame != _editData->selectedGeneIndex && !_selectionChangedFromTree;
        _selectedGeneFromPreviousFrame = _editData->selectedGeneIndex;
        _selectionChangedFromTree = false;

        auto rootHull = GenomeDescInfoService::get().getReferencedGenesInRootGeneHull(_editData->genome);
        auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;

        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTable("Structure list", 4, flags, ImVec2(-1, -1), 0.0f)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, scale(150.0f));
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, scale(130.0f));
            ImGui::TableSetupColumn("References", ImGuiTableColumnFlags_WidthFixed, scale(95.0f));
            ImGui::TableSetupColumn("Referenced by", ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();

            for (auto const& [index, gene] : _editData->genome._genes | boost::adaptors::indexed(0)) {
                auto geneIndex = toInt(index);
                processGeneNode(geneIndex, gene, !rootHull.contains(geneIndex), scrollToSelection, customizationColors);
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

namespace
{
    std::string toIndexList(std::vector<int> const& indices)
    {
        auto strings = indices | std::views::transform([](auto const& index) { return std::to_string(index); });
        return boost::algorithm::join(std::vector(strings.begin(), strings.end()), ", ");
    }
}

void _GenomeEditorWidget::processGeneNode(
    int geneIndex,
    GeneDesc const& gene,
    bool isUnreachable,
    bool scrollToSelection,
    ColorVector<FloatColorRGB> const& customizationColors)
{
    ImGui::PushID(geneIndex);
    ImGui::TableNextRow();

    auto isSelectedGene = _editData->selectedGeneIndex == geneIndex;
    auto flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_DefaultOpen;
    if (isSelectedGene && _editData->selectionLevel == GenomeSelectionLevel::Gene) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (scrollToSelection && isSelectedGene) {
        ImGui::SetNextItemOpen(true);
    }

    // Column 0: identity, followed by the optional name in decent parentheses
    ImGui::TableNextColumn();
    if (isUnreachable) {
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextConflictColor.Value);
    }
    auto isOpen = ImGui::TreeNodeEx("##gene", flags, "Gene %d", geneIndex);
    if (isUnreachable) {
        ImGui::PopStyleColor();
    }
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        _editData->selectGene(geneIndex);
        _selectionChangedFromTree = true;
    }
    if (!gene._name.empty()) {
        ImGui::SameLine();
        AlienGui::Text(AlienGui::TextParameters().text("(" + gene._name + ")").style(AlienGui::TextStyle::Decent));
    }
    if (scrollToSelection && isSelectedGene) {
        ImGui::SetScrollHereY();
    }

    // Column 1: the customization colors in use, followed by node count and shape
    ImGui::TableNextColumn();
    std::set<int> usedColors;
    for (auto const& node : gene._nodes) {
        usedColors.insert(node._color);
    }
    for (auto const& usedColor : usedColors) {
        float h, s, v;
        AlienGui::ConvertRGBtoHSV(customizationColors.values[usedColor].toRgbColor(), h, s, v);
        AlienGui::Chip(AlienGui::ChipParameters().dotColor(ImColor::HSV(h, s, v)).dotSize(ColorChipSize).spacing(ColorChipSpacing));
    }
    AlienGui::Text(std::to_string(gene._nodes.size()) + "-" + Const::ConstructorShapeStrings.at(gene._shape));

    // Column 2: referenced genes
    ImGui::TableNextColumn();
    AlienGui::Text(toIndexList(GenomeDescInfoService::get().getReferences(gene)));

    // Column 3: referencing genes
    ImGui::TableNextColumn();
    auto referencedBy = toIndexList(GenomeDescInfoService::get().getReferencedBy(_editData->genome, geneIndex));
    if (referencedBy.empty() && geneIndex > 0) {
        referencedBy = "-";
    }
    AlienGui::Text(referencedBy);

    if (isOpen) {
        for (auto const& [index, node] : gene._nodes | boost::adaptors::indexed(0)) {
            processNodeLeaf(geneIndex, toInt(index), gene, node, customizationColors);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void _GenomeEditorWidget::processNodeLeaf(
    int geneIndex,
    int nodeIndex,
    GeneDesc const& gene,
    NodeDesc const& node,
    ColorVector<FloatColorRGB> const& customizationColors)
{
    ImGui::PushID(nodeIndex);
    ImGui::TableNextRow();

    auto flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAllColumns;
    if (_editData->selectionLevel == GenomeSelectionLevel::Node && _editData->selectedGeneIndex == geneIndex
        && _editData->getSelectedNodeIndex() == nodeIndex) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }

    // Column 0: identity
    ImGui::TableNextColumn();
    ImGui::TreeNodeEx("##node", flags, "Node %d", nodeIndex);
    if (ImGui::IsItemClicked()) {
        _editData->selectNode(geneIndex, nodeIndex);
        _selectionChangedFromTree = true;
    }

    // Column 1: cell type. With homogeneous cell type every node shows the cell type of the first node
    ImGui::TableNextColumn();
    float h, s, v;
    AlienGui::ConvertRGBtoHSV(customizationColors.values[node._color].toRgbColor(), h, s, v);
    AlienGui::Chip(AlienGui::ChipParameters().dotColor(ImColor::HSV(h, s, v)).dotSize(ColorChipSize).spacing(ColorChipSpacing));
    auto nodeType = gene._homogeneousCellType ? gene._nodes.front().getCellType() : node.getCellType();
    AlienGui::Text(Const::CellTypeStrings.at(nodeType));

    // Column 2: the gene this node constructs
    ImGui::TableNextColumn();
    if (node._constructor.has_value()) {
        auto constructedGeneIndex = node._constructor->_geneIndex;
        auto text = "Gene " + std::to_string(constructedGeneIndex);
        auto const& genes = _editData->genome._genes;
        if (_editData->hasValidGeneIndex(constructedGeneIndex) && !genes.at(constructedGeneIndex)._name.empty()) {
            text += ": " + genes.at(constructedGeneIndex)._name;
        }
        AlienGui::Text(text);
    }

    // Column 3: referencing genes -- only genes have them
    ImGui::TableNextColumn();

    ImGui::PopID();
}

void _GenomeEditorWidget::processStructureButtons()
{
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_CIRCLE).tooltip("Add gene"))) {
        onAddGene();
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value());
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_SQUARE).tooltip("Add node to the selected gene"))) {
        onAddNode();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    auto nodeLevel = _editData->isNodeLevelSelected();
    ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value());
    if (AlienGui::ActionButton(
            AlienGui::ActionButtonParameters().buttonText(ICON_FA_MINUS_CIRCLE).tooltip(nodeLevel ? "Remove selected node" : "Remove selected gene"))) {
        if (nodeLevel) {
            onRemoveNode();
        } else {
            onRemoveGene();
        }
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    nodeLevel = _editData->isNodeLevelSelected();
    auto canMoveUpward = nodeLevel ? _editData->getSelectedNodeIndex().value() > 0 : _editData->selectedGeneIndex.value_or(0) > 0;
    ImGui::BeginDisabled(!canMoveUpward);
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(ICON_FA_CHEVRON_CIRCLE_UP)
                                   .tooltip(nodeLevel ? "Move selected node upward" : "Move selected gene upward"))) {
        if (nodeLevel) {
            onMoveNodeUpward();
        } else {
            onMoveGeneUpward();
        }
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    nodeLevel = _editData->isNodeLevelSelected();
    auto canMoveDownward = false;
    if (nodeLevel) {
        canMoveDownward = _editData->getSelectedNodeIndex().value() + 1 < toInt(_editData->getSelectedGeneRef()._nodes.size());
    } else if (_editData->selectedGeneIndex.has_value()) {
        canMoveDownward = _editData->selectedGeneIndex.value() + 1 < toInt(_editData->genome._genes.size());
    }
    ImGui::BeginDisabled(!canMoveDownward);
    if (AlienGui::ActionButton(AlienGui::ActionButtonParameters()
                                   .buttonText(ICON_FA_CHEVRON_CIRCLE_DOWN)
                                   .tooltip(nodeLevel ? "Move selected node downward" : "Move selected gene downward"))) {
        if (nodeLevel) {
            onMoveNodeDownward();
        } else {
            onMoveGeneDownward();
        }
    }
    ImGui::EndDisabled();
}

void _GenomeEditorWidget::onAddGene()
{
    auto& genome = _editData->genome;
    auto name = "Gene " + std::to_string(_sequenceNumberForCreatedGenes++);
    if (genome._genes.empty()) {
        auto newGene = GeneDesc().name(name).nodes({NodeDesc()}).shape(ConstructorShape_Segment);
        GenomeDescEditService::get().addGene(genome, 0, newGene);
        _editData->selectGene(0);
    } else {
        int insertIndex;
        if (_editData->selectedGeneIndex.has_value()) {
            insertIndex = _editData->selectedGeneIndex.value();
        } else {
            insertIndex = toInt(genome._genes.size()) - 1;
        }

        auto newGene = GeneDesc().name(name).nodes({NodeDesc()}).shape(ConstructorShape_Segment);
        GenomeDescEditService::get().addGene(genome, insertIndex, newGene);

        // Adapt gene selection
        _editData->selectGene(insertIndex + 1);

        // Adapt node selection
        std::map<int, int> newSelectedNodeByGeneIndex;
        for (auto const& [index, selectedNode] : _editData->selectedNodeByGeneIndex) {
            if (index <= insertIndex) {
                newSelectedNodeByGeneIndex.emplace(index, selectedNode);
            } else {
                newSelectedNodeByGeneIndex.emplace(index + 1, selectedNode);
            }
        }
        _editData->selectedNodeByGeneIndex = newSelectedNodeByGeneIndex;
        _editData->setSelectedNodeIndex(0);
    }
}

void _GenomeEditorWidget::onAddNode()
{
    auto& gene = _editData->getSelectedGeneRef();
    auto selectedNode = _editData->getSelectedNodeIndex();
    auto geneIndex = _editData->selectedGeneIndex.value();
    if (gene._nodes.empty()) {
        GenomeDescEditService::get().addNode(gene, 0, NodeDesc());
        _editData->selectNode(geneIndex, 0);
    } else {
        int insertIndex = selectedNode.has_value() ? selectedNode.value() : toInt(gene._nodes.size()) - 1;
        auto insertAtBack = insertIndex == toInt(gene._nodes.size()) - 1;

        auto& nodeAtInsertIndex = gene._nodes.at(insertIndex);
        int color = nodeAtInsertIndex._color;

        auto newNode = NodeDesc().color(color);
        if (insertAtBack) {
            nodeAtInsertIndex._neuralNetwork._connectionWeights.at(0) = 1;
            newNode._neuralNetwork._connectionWeights.at(0) = 0;
        }
        GenomeDescEditService::get().addNode(gene, insertIndex, newNode);

        _editData->selectNode(geneIndex, insertIndex + 1);
    }
}

void _GenomeEditorWidget::onRemoveNode()
{
    int removeIndex = _editData->getSelectedNodeIndex().value();
    auto& gene = _editData->getSelectedGeneRef();

    GenomeDescEditService::get().removeNode(gene, removeIndex);

    // Adapt node selection
    auto& nodes = gene._nodes;
    if (nodes.empty()) {
        _editData->setSelectedNodeIndex(std::nullopt);
        _editData->selectionLevel = GenomeSelectionLevel::Gene;
    } else if (removeIndex >= toInt(nodes.size())) {
        _editData->setSelectedNodeIndex(toInt(nodes.size()) - 1);
    } else {
        _editData->setSelectedNodeIndex(removeIndex);
    }
}

void _GenomeEditorWidget::onMoveNodeUpward()
{
    auto indexToMove = _editData->getSelectedNodeIndex().value();
    auto& gene = _editData->getSelectedGeneRef();

    if (indexToMove == 1 && gene._nodes.at(indexToMove).getCellType() == CellType_Void) {
        showMessage("Error", "The first node cannot be void.");
        return;
    }

    GenomeDescEditService::get().swapNodes(gene, indexToMove - 1);
    _editData->setSelectedNodeIndex(indexToMove - 1);
}

void _GenomeEditorWidget::onMoveNodeDownward()
{
    auto indexToMove = _editData->getSelectedNodeIndex().value();
    auto& gene = _editData->getSelectedGeneRef();

    if (indexToMove == gene._nodes.size() - 2 && gene._nodes.at(indexToMove).getCellType() == CellType_Void) {
        showMessage("Error", "The last node cannot be void.");
        return;
    }

    GenomeDescEditService::get().swapNodes(gene, indexToMove);
    _editData->setSelectedNodeIndex(indexToMove + 1);
}

void _GenomeEditorWidget::onRemoveGene()
{
    auto referencedBy = GenomeDescInfoService::get().getReferencedBy(_editData->genome, _editData->selectedGeneIndex.value());
    if (!referencedBy.empty()) {
        auto referencedByStrings = referencedBy | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex); });
        auto referencedByString = boost::algorithm::join(std::vector(referencedByStrings.begin(), referencedByStrings.end()), ", ");
        auto text = referencedBy.size() == 1 ? "This gene could not be removed since it is still used by gene "
                                             : "This gene could not be removed since it is still used by genes ";
        GenericMessageDialog::get().information("Error", text + referencedByString + ".");
        return;
    }
    if (_editData->selectedGeneIndex.value() == 0) {
        GenericMessageDialog::get().yesNo(
            "Delete root gene",
            "Do you really want to delete the root gene? If you decide to do so, the following gene will become the new root gene.",
            [this] { this->removeGeneIntern(); });
        return;
    }
    removeGeneIntern();
}

void _GenomeEditorWidget::onMoveGeneUpward()
{
    if (_editData->selectedGeneIndex.value() == 1) {
        GenericMessageDialog::get().yesNo("Swap root gene", "Do you really want to swap the root gene?", [this] { this->moveGeneUpwardIntern(); });
        return;
    }
    moveGeneUpwardIntern();
}

void _GenomeEditorWidget::onMoveGeneDownward()
{
    if (_editData->selectedGeneIndex.value() == 0) {
        GenericMessageDialog::get().yesNo("Swap root gene", "Do you really want to swap the root gene?", [this] { this->moveGeneDownwardIntern(); });
        return;
    }
    moveGeneDownwardIntern();
}

void _GenomeEditorWidget::removeGeneIntern()
{
    int removeIndex = _editData->selectedGeneIndex.value();

    GenomeDescEditService::get().removeGene(_editData->genome, removeIndex);

    // Adapt gene selection
    auto& genes = _editData->genome._genes;
    if (genes.empty()) {
        _editData->selectedGeneIndex.reset();
        _editData->selectionLevel = GenomeSelectionLevel::Gene;
    } else if (removeIndex >= toInt(genes.size())) {
        _editData->selectGene(toInt(genes.size()) - 1);
    } else {
        _editData->selectGene(removeIndex);
    }

    // Adapt node selection
    std::map<int, int> newSelectedNodeByGeneIndex;
    for (auto const& [index, selectedNode] : _editData->selectedNodeByGeneIndex) {
        if (index < removeIndex) {
            newSelectedNodeByGeneIndex.emplace(index, selectedNode);
        } else if (index > removeIndex) {
            newSelectedNodeByGeneIndex.emplace(index - 1, selectedNode);
        }
    }
    _editData->selectedNodeByGeneIndex = newSelectedNodeByGeneIndex;
}

void _GenomeEditorWidget::moveGeneUpwardIntern()
{
    int indexToMove = _editData->selectedGeneIndex.value();
    GenomeDescEditService::get().swapGenes(_editData->genome, indexToMove - 1);

    // Adapt gene selection
    --_editData->selectedGeneIndex.value();

    // Adapt node selection
    std::map<int, int> newSelectedNodeByGeneIndex;
    for (auto const& [index, selectedNode] : _editData->selectedNodeByGeneIndex) {
        if (index == indexToMove) {
            newSelectedNodeByGeneIndex.emplace(index - 1, selectedNode);
        } else if (index == indexToMove - 1) {
            newSelectedNodeByGeneIndex.emplace(index + 1, selectedNode);
        } else {
            newSelectedNodeByGeneIndex.emplace(index, selectedNode);
        }
    }
    _editData->selectedNodeByGeneIndex = newSelectedNodeByGeneIndex;
}

void _GenomeEditorWidget::moveGeneDownwardIntern()
{
    int indexToMove = _editData->selectedGeneIndex.value();
    GenomeDescEditService::get().swapGenes(_editData->genome, indexToMove);

    // Adapt gene selection
    ++_editData->selectedGeneIndex.value();

    // Adapt node selection
    std::map<int, int> newSelectedNodeByGeneIndex;
    for (auto const& [index, selectedNode] : _editData->selectedNodeByGeneIndex) {
        if (index == indexToMove) {
            newSelectedNodeByGeneIndex.emplace(index + 1, selectedNode);
        } else if (index == indexToMove + 1) {
            newSelectedNodeByGeneIndex.emplace(index - 1, selectedNode);
        } else {
            newSelectedNodeByGeneIndex.emplace(index, selectedNode);
        }
    }
    _editData->selectedNodeByGeneIndex = newSelectedNodeByGeneIndex;
}
