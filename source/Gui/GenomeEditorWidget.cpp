#include "GenomeEditorWidget.h"

#include <ranges>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"

#include "AlienGui.h"
#include "CreatureTabEditData.h"
#include "CreatureTabLayoutData.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
}


GenomeEditorWidget _GenomeEditorWidget::create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
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

_GenomeEditorWidget::_GenomeEditorWidget(CreatureTabEditData const& genome, CreatureTabLayoutData const& layoutData)
    : _editData(genome)
    , _layoutData(layoutData)
{}

void _GenomeEditorWidget::processHeaderData()
{
    AlienGui::Group("Genome");

    auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));
    if (ImGui::BeginChild("GenomeHeader", ImVec2(0, -_layoutData->geneListHeight), 0)) {

        auto numNodesString = std::to_string(GenomeDescriptionInfoService::get().getNumberOfNodes(_editData->genome));
        AlienGui::InputText(AlienGui::InputTextParameters().name("Node count").readOnly(true).textWidth(rightColumnWidth), numNodesString);

        auto numCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(_editData->genome);
        auto numCellsString = numCells != -1 ? std::to_string(numCells) : std::string("infinity");
        AlienGui::InputText(AlienGui::InputTextParameters().name("Resulting cells").readOnly(true).textWidth(rightColumnWidth), numCellsString);

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Front angle").format("%.1f").textWidth(rightColumnWidth), _editData->genome._frontAngle);
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::processGeneList()
{
    if (ImGui::BeginChild("GeneList", ImVec2(0, 0))) {
        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTable("Gene list", 6, flags, ImVec2(-1, -1), 0.0f)) {
            ImGui::TableSetupColumn("No.", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(30.0f));
            ImGui::TableSetupColumn("Gene type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(80.0f));
            ImGui::TableSetupColumn("Node count", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(90.0f));
            ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
            ImGui::TableSetupColumn("References", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(120.0f));
            ImGui::TableSetupColumn("Referenced by", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(120.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

            auto const& genome = _editData->genome;

            ImGuiListClipper clipper;
            clipper.Begin(genome._genes.size());
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    auto const& gene = genome._genes.at(row);

                    ImGui::PushID(row);
                    ImGui::TableNextRow(0, scale(21.0f));

                    // Column 0: No.
                    ImGui::TableNextColumn();
                    AlienGui::Text(std::to_string(row + 1));
                    ImGui::SameLine();
                    auto selected = _editData->selectedGeneIndex.has_value() ? _editData->selectedGeneIndex.value() == row : false;
                    if (ImGui::Selectable(
                            "",
                            &selected,
                            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                            ImVec2(0, scale(ImGui::GetTextLineHeightWithSpacing()) - ImGui::GetStyle().FramePadding.y))) {
                        if (selected) {
                            _editData->selectedGeneIndex = row;
                        }
                    }

                    // Column 1: Gene type
                    ImGui::TableNextColumn();
                    if (row == 0) {
                        AlienGui::Text("Principal");
                    } else {
                        AlienGui::Text("Auxiliary");
                    }

                    // Column 2: Node count
                    ImGui::TableNextColumn();
                    AlienGui::Text(std::to_string(gene._nodes.size()));

                    // Column 3: Shape
                    ImGui::TableNextColumn();
                    AlienGui::Text(Const::ConstructionShapeStrings.at(gene._shape));

                    // Column 4: References
                    ImGui::TableNextColumn();
                    auto references = GenomeDescriptionInfoService::get().getReferences(gene);
                    auto referencesStrings = references | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                    auto referencesString = boost::algorithm::join(std::vector(referencesStrings.begin(), referencesStrings.end()), ", ");
                    AlienGui::Text(referencesString);

                    // Column 5: Referenced by
                    ImGui::TableNextColumn();
                    auto referencedBy = GenomeDescriptionInfoService::get().getReferencedBy(genome, row);
                    if (!referencedBy.empty()) {
                        auto referencedByStrings = referencedBy | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                        auto referencedByString = boost::algorithm::join(std::vector(referencedByStrings.begin(), referencedByStrings.end()), ", ");
                        AlienGui::Text(referencedByString);
                    } else {
                        AlienGui::Text("Unused");
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

    ImVec2 buttonGroupSize = {scale(108.0f), scale(22.0f)};
    ImGui::SetCursorScreenPos(
        ImVec2(cursorPos.x + ImGui::GetContentRegionAvail().x - buttonGroupSize.x - scale(15.0f), cursorPos.y - buttonGroupSize.y - scale(20.0f)));
    if (ImGui::BeginChild("ButtonGroup", buttonGroupSize)) {

        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_CIRCLE))) {
            onAddGene();
        }
        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value());
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_MINUS_CIRCLE))) {
            onRemoveGene();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value() || _editData->selectedGeneIndex.value() == 0);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_UP))) {
            onMoveGeneUpward();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!_editData->selectedGeneIndex.has_value() || _editData->selectedGeneIndex.value() == _editData->genome._genes.size() - 1);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_DOWN))) {
            onMoveGeneDownward();
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::onAddGene()
{
    auto& genome = _editData->genome;
    if (genome._genes.empty()) {
        GenomeDescriptionEditService::get().addEmptyGene(genome, 0);
        _editData->selectedGeneIndex = 0;
    } else {
        int insertIndex;
        if (_editData->selectedGeneIndex.has_value()) {
            insertIndex = _editData->selectedGeneIndex.value();
        } else {
            insertIndex = toInt(genome._genes.size()) - 1;
        }

        GenomeDescriptionEditService::get().addEmptyGene(genome, insertIndex);

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
    }
}

void _GenomeEditorWidget::onRemoveGene()
{
    auto referencedBy = GenomeDescriptionInfoService::get().getReferencedBy(_editData->genome, _editData->selectedGeneIndex.value());
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
            "Delete principal gene",
            "Do you really want to delete the principal gene? If you decide to do so, the following gene will become the new principal gene.",
            [this] { this->removeGeneIntern(); });
        return;
    }
    removeGeneIntern();
}

void _GenomeEditorWidget::onMoveGeneUpward()
{
    if (_editData->selectedGeneIndex.value() == 1) {
        GenericMessageDialog::get().yesNo("Swap principal gene", "Do you really want to swap the principal gene?", [this] { this->moveGeneUpwardIntern(); });
        return;
    }
    moveGeneUpwardIntern();
}

void _GenomeEditorWidget::onMoveGeneDownward()
{
    if (_editData->selectedGeneIndex.value() == 0) {
        GenericMessageDialog::get().yesNo("Swap principal gene", "Do you really want to swap the principal gene?", [this] { this->moveGeneDownwardIntern(); });
        return;
    }
    moveGeneDownwardIntern();
}

void _GenomeEditorWidget::removeGeneIntern()
{
    int removeIndex = _editData->selectedGeneIndex.value();

    GenomeDescriptionEditService::get().removeGene(_editData->genome, removeIndex);

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
        } else {
            newSelectedNodeByGeneIndex.emplace(index - 1, selectedNode);
        }
    }
    _editData->selectedNodeByGeneIndex = newSelectedNodeByGeneIndex;
}

void _GenomeEditorWidget::moveGeneUpwardIntern()
{
    int indexToMove = _editData->selectedGeneIndex.value();
    GenomeDescriptionEditService::get().swapGenes(_editData->genome, indexToMove - 1);

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
    GenomeDescriptionEditService::get().swapGenes(_editData->genome, indexToMove);

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
