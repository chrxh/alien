#include "GenomeEditorWidget.h"

#include <ranges>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <EngineInterface/GenomeDescEditService.h>
#include <EngineInterface/GenomeDescInfoService.h>

#include "AlienGui.h"
#include "GenericMessageDialog.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "MutationRatesDialog.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
    auto constexpr HeaderMinColumnWidth = 300.0f;
}


GenomeEditorWidget _GenomeEditorWidget::create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
{
    return GenomeEditorWidget(new _GenomeEditorWidget(editData, layoutData));
}

void _GenomeEditorWidget::process()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_layoutData->genomeEditorWidth, 0))) {
        processHeaderData();

        AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->geneListHeight);

        processGeneList();
        processGeneListButtons();
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
                _sequenceNumberForCreatedGenes = std::max(_sequenceNumberForCreatedGenes, number);
            }
        } catch (...) {
        }
    }
}

void _GenomeEditorWidget::processHeaderData()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Genome").highlighted(true));

    if (ImGui::BeginChild("GenomeHeader", ImVec2(0, -_layoutData->geneListHeight), 0)) {

        AlienGui::DynamicTableLayout table(HeaderMinColumnWidth);
        if (table.begin()) {
            auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

            AlienGui::Group(AlienGui::GroupParameters().text("Base properties and info"));
            AlienGui::InputText(AlienGui::InputTextParameters().name("Genome name").textWidth(rightColumnWidth), _editData->genome._name);
            if (AlienGui::InputInt(AlienGui::InputIntParameters().name("Lineage").textWidth(rightColumnWidth), _editData->genome._lineageId)) {
                _editData->genome._prevLineageId.reset();
            }

            auto numNodesString = std::to_string(GenomeDescInfoService::get().getNumberOfNodes(_editData->genome));
            AlienGui::InputText(AlienGui::InputTextParameters().name("Node count").readOnly(true).textWidth(rightColumnWidth), numNodesString);

            auto numCells = GenomeDescInfoService::get().getNumberOfResultingCells(_editData->genome);
            auto numCellsString = numCells != -1 ? std::to_string(numCells) : std::string("Infinity");
            AlienGui::InputText(AlienGui::InputTextParameters().name("Resulting cells").readOnly(true).textWidth(rightColumnWidth), numCellsString);

            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Front angle").format("%.1f").min(-180.0f).max(180.0f).textWidth(rightColumnWidth),
                &_editData->genome._frontAngle);

            AlienGui::Checkbox(
                AlienGui::CheckboxParameters()
                    .name("Resistance to injection")
                    .textWidth(rightColumnWidth)
                    .tooltip("If enabled, no cell of a creature with this genome can be injected by a hostile injector."),
                _editData->genome._resistanceToInjection);

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Mutation rates"));

            auto buttonWidth = scale(60.0f);
            auto availableWidth = ImGui::GetContentRegionAvail().x;
            auto listBoxWidth = availableWidth - buttonWidth - ImGui::GetStyle().ItemSpacing.x;
            AlienGui::ListBox(AlienGui::ListBoxParameters().items(_editData->genome._mutationRates.getActiveMutationTypes()).width(listBoxWidth));
            ImGui::SameLine();
            if (AlienGui::Button("Edit")) {
                MutationRatesDialog::get().open(
                    _editData->genome._mutationRates, [this](MutationRatesDesc const& mutationRates) { _editData->genome._mutationRates = mutationRates; });
            }

            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters()
                    .name("Accumulated mutations")
                    .format("%.5f")
                    .min(0.0f)
                    .max(10000.0f)
                    .logarithmic(true)
                    .textWidth(rightColumnWidth),
                &_editData->genome._accumulatedMutations);

            table.next();

            table.end();
        }
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::processGeneList()
{
    if (ImGui::BeginChild("GeneList", ImVec2(0, 0))) {
        auto scrollToGeneIndex = -1;
        if (!_selectedGeneFromPreviousFrame.has_value() || _selectedGeneFromPreviousFrame != _editData->selectedGeneIndex) {
            if (_editData->selectedGeneIndex.has_value() && !_geneSelectedFromTable) {
                scrollToGeneIndex = std::max(1, _editData->selectedGeneIndex.value());
            }
        }
        _selectedGeneFromPreviousFrame = _editData->selectedGeneIndex;
        _geneSelectedFromTable = false;

        auto rootHull = GenomeDescInfoService::get().getReferencedGenesInRootGeneHull(_editData->genome);

        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTable("Gene list", 6, flags, ImVec2(-1, -1), 0.0f)) {
            ImGui::TableSetupColumn("Gene", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(105.0f));
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(120.0f));
            ImGui::TableSetupColumn("References", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(80.0f));
            ImGui::TableSetupColumn("Referenced by", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(95.0f));
            ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(60.0f));
            ImGui::TableSetupColumn("Nodes", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(70.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

            auto const& genome = _editData->genome;

            ImGuiListClipper clipper;
            clipper.Begin(genome._genes.size());
            if (scrollToGeneIndex != -1) {
                clipper.IncludeItemByIndex(scrollToGeneIndex);
            }
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    auto const& gene = genome._genes.at(row);
                    if (row == scrollToGeneIndex) {
                        ImGui::SetScrollHereY();
                    }

                    ImGui::PushID(row);
                    ImGui::TableNextRow(0, scale(21.0f));

                    auto isUnreachable = !rootHull.contains(row);
                    if (isUnreachable) {
                        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::TextConflictColor);
                    }

                    // Column 0: No.
                    ImGui::TableNextColumn();
                    AlienGui::Text(std::to_string(row + 1));
                    if (row == 0) {
                        ImGui::SameLine();
                        AlienGui::Text(AlienGui::TextParameters().text(" (root)").style(AlienGui::TextStyle::Decent));
                    }
                    ImGui::SameLine();
                    auto selected = _editData->selectedGeneIndex.has_value() ? _editData->selectedGeneIndex.value() == row : false;
                    if (ImGui::Selectable(
                            "",
                            &selected,
                            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                            ImVec2(0, scale(ImGui::GetTextLineHeightWithSpacing()) - ImGui::GetStyle().FramePadding.y))) {
                        if (selected) {
                            _geneSelectedFromTable = true;
                            _editData->selectedGeneIndex = row;
                        }
                    }

                    // Column 1: Name
                    ImGui::TableNextColumn();
                    if (!gene._name.empty()) {
                        AlienGui::Text(gene._name);
                    } else {
                        AlienGui::Text(AlienGui::TextParameters().text("(unnamed)").style(AlienGui::TextStyle::Decent));
                    }

                    // Column 2: References
                    ImGui::TableNextColumn();
                    auto references = GenomeDescInfoService::get().getReferences(gene);
                    auto referencesStrings = references | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                    auto referencesString = boost::algorithm::join(std::vector(referencesStrings.begin(), referencesStrings.end()), ", ");
                    AlienGui::Text(referencesString);

                    // Column 3: Referenced by
                    ImGui::TableNextColumn();
                    auto referencedBy = GenomeDescInfoService::get().getReferencedBy(genome, row);
                    if (!referencedBy.empty()) {
                        auto referencedByStrings = referencedBy | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                        auto referencedByString = boost::algorithm::join(std::vector(referencedByStrings.begin(), referencedByStrings.end()), ", ");
                        AlienGui::Text(referencedByString);
                    } else {
                        if (row > 0) {
                            AlienGui::Text("-");
                        }
                    }

                    // Column 4: Shape
                    ImGui::TableNextColumn();
                    AlienGui::Text(Const::ConstructorShapeStrings.at(gene._shape));

                    // Column 5: Node count
                    ImGui::TableNextColumn();
                    AlienGui::Text(std::to_string(gene._nodes.size()));

                    if (isUnreachable) {
                        ImGui::PopStyleColor();
                    }

                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::processGeneListButtons()
{
    auto cursorPos = ImGui::GetCursorScreenPos();

    ImVec2 buttonGroupSize = {scale(122.0f), scale(36.0f)};
    ImGui::SetCursorScreenPos(
        ImVec2(cursorPos.x + ImGui::GetContentRegionAvail().x - buttonGroupSize.x - scale(15.0f), cursorPos.y - buttonGroupSize.y - scale(20.0f)));
    if (ImGui::BeginChild("ButtonGroup", buttonGroupSize)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        ImGui::GetWindowDrawList()->AddRectFilled({startPos.x, startPos.y}, {startPos.x + size.x, startPos.y + size.y}, ImColor::HSV(0.0f, 0.0f, 0.055f, 0.5f));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + scale(7.0f));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + scale(7.0f));

        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_CIRCLE).frame(true).transparentBackground(false))) {
            onAddGene();
        }
        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value());
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_MINUS_CIRCLE).frame(true).transparentBackground(false))) {
            onRemoveGene();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value() || _editData->selectedGeneIndex.value() == 0);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_UP).frame(true).transparentBackground(false))) {
            onMoveGeneUpward();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value() || _editData->selectedGeneIndex.value() == _editData->genome._genes.size() - 1);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_DOWN).frame(true).transparentBackground(false))) {
            onMoveGeneDownward();
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::onAddGene()
{
    auto& genome = _editData->genome;
    auto name = "Gene " + std::to_string(++_sequenceNumberForCreatedGenes);
    if (genome._genes.empty()) {
        auto newGene = GeneDesc().name(name).nodes({NodeDesc()}).shape(ConstructorShape_Segment);
        GenomeDescEditService::get().addGene(genome, 0, newGene);
        _editData->selectedGeneIndex = 0;
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
        _editData->selectedGeneIndex = insertIndex + 1;

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

void _GenomeEditorWidget::onRemoveGene()
{
    auto referencedBy = GenomeDescInfoService::get().getReferencedBy(_editData->genome, _editData->selectedGeneIndex.value());
    if (!referencedBy.empty()) {
        auto referencedByStrings = referencedBy | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
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
    } else if (removeIndex >= toInt(genes.size())) {
        _editData->selectedGeneIndex = toInt(genes.size()) - 1;
    } else {
        _editData->selectedGeneIndex = removeIndex;
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
