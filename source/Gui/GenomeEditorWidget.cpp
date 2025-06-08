#include "GenomeEditorWidget.h"

#include <ranges>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"

#include "AlienImGui.h"
#include "CreatureTabGenomeData.h"
#include "CreatureTabLayoutData.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderTotalWidth = 400.0f;
    auto constexpr HeaderRightColumnWidth = 120.0f;
}

GenomeEditorWidget _GenomeEditorWidget::create(CreatureTabGenomeData const& editData, CreatureTabLayoutData const& layoutData)
{
    return GenomeEditorWidget(new _GenomeEditorWidget(editData, layoutData));
}

void _GenomeEditorWidget::process()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_layoutData->genomeEditorWidth, 0))) {
        processHeaderData();

        AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->geneListHeight);

        processGeneList();
        processGeneListButtons();
    }
    ImGui::EndChild();
}

_GenomeEditorWidget::_GenomeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _editData(genome)
    , _layoutData(layoutData)
{}

void _GenomeEditorWidget::processHeaderData()
{
    AlienImGui::Group("Genome");

    auto availableWidth = scaleInverse(ImGui::GetContentRegionAvail().x);
    auto width = scale(std::min(availableWidth, HeaderTotalWidth));
    if (ImGui::BeginChild("GenomeHeader", ImVec2(width, ImGui::GetContentRegionAvail().y - _layoutData->geneListHeight), 0)) {
        auto numNodesString = std::to_string(GenomeDescriptionInfoService::get().getNumberOfNodes(_editData->genome));
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Node count").readOnly(true).textWidth(HeaderRightColumnWidth), numNodesString);
        auto numCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(_editData->genome);
        auto numCellsString = numCells != -1 ? std::to_string(numCells) : std::string("Infinity");
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Resulting cells").readOnly(true).textWidth(HeaderRightColumnWidth), numCellsString);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Front angle").format("%.1f").textWidth(HeaderRightColumnWidth), _editData->genome._frontAngle);
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
            ImGui::TableSetupColumn("Referenced genes", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(120.0f));
            ImGui::TableSetupColumn("Referencing genes", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(120.0f));
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
                    AlienImGui::Text(std::to_string(row + 1));
                    ImGui::SameLine();
                    auto selected = _editData->selectedGene.has_value() ? _editData->selectedGene == row : false;
                    if (ImGui::Selectable(
                            "",
                            &selected,
                            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                            ImVec2(0, scale(ImGui::GetTextLineHeightWithSpacing()) - ImGui::GetStyle().FramePadding.y))) {
                        if (selected) {
                            _editData->selectedGene = row;
                        }
                    }
                    // Column 1: Gene type
                    ImGui::TableNextColumn();
                    if (row == 0) {
                        AlienImGui::Text("Principal");
                    } else {
                        AlienImGui::Text("Auxiliary");
                    }

                    // Column 2: Node count
                    ImGui::TableNextColumn();
                    AlienImGui::Text(std::to_string(gene._nodes.size()));

                    // Column 3: Shape
                    ImGui::TableNextColumn();
                    AlienImGui::Text(Const::ConstructionShapeNames.at(gene._shape));

                    // Column 4: Referenced genes
                    ImGui::TableNextColumn();
                    auto referencedGenes = GenomeDescriptionInfoService::get().getReferencedGeneIndices(gene);
                    auto referencedGenesStrings = referencedGenes | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                    auto referencedGenesString = boost::algorithm::join(std::vector(referencedGenesStrings.begin(), referencedGenesStrings.end()), ", ");
                    AlienImGui::Text(referencedGenesString);

                    // Column 5: Referencing genes
                    ImGui::TableNextColumn();
                    auto referencingGenes = GenomeDescriptionInfoService::get().getReferencingGeneIndices(genome, row);
                    auto referencingGenesStrings =
                        referencingGenes | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                    auto referencingGenesString = boost::algorithm::join(std::vector(referencingGenesStrings.begin(), referencingGenesStrings.end()), ", ");
                    AlienImGui::Text(referencingGenesString);

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

    ImVec2 buttonGroupSize = {scale(105.0f), scale(22.0f)};
    ImGui::SetCursorScreenPos(
        ImVec2(cursorPos.x + ImGui::GetContentRegionAvail().x - buttonGroupSize.x - scale(10.0f), cursorPos.y - buttonGroupSize.y - scale(20.0f)));
    if (ImGui::BeginChild("ButtonGroup", buttonGroupSize)) {

        if (AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_CIRCLE))) {
            onAddGene();
        }
        ImGui::SameLine();
        AlienImGui::PaddingLeft();
        ImGui::BeginDisabled(!_editData->selectedGene.has_value() || _editData->selectedGene.value() == 0);
        if (AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_MINUS_CIRCLE))) {
            onRemoveGene();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienImGui::PaddingLeft();
        ImGui::BeginDisabled(!_editData->selectedGene.has_value() || _editData->selectedGene.value() == 0);
        if (AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_UP))) {
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienImGui::PaddingLeft();
        ImGui::BeginDisabled(!_editData->selectedGene.has_value() || _editData->selectedGene.value() == 0);
        if (AlienImGui::ActionButton(AlienImGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_DOWN))) {
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
        _editData->selectedGene = 0;
    } else {
        int insertIndex;
        if (_editData->selectedGene.has_value()) {
            insertIndex = _editData->selectedGene.value() + 1;
        } else {
            insertIndex = static_cast<int>(genome._genes.size());
        }

        GenomeDescriptionEditService::get().addEmptyGene(genome, insertIndex);

        _editData->selectedGene = insertIndex;
    }
}

void _GenomeEditorWidget::onRemoveGene()
{
    if (!_editData->selectedGene.has_value()) {
        return;
    }
    int removeIndex = _editData->selectedGene.value();
    auto& genes = _editData->genome._genes;

    GenomeDescriptionEditService::get().removeGene(_editData->genome, removeIndex);

    if (genes.empty()) {
        _editData->selectedGene.reset();
    } else if (removeIndex >= toInt(genes.size())) {
        _editData->selectedGene = toInt(genes.size()) - 1;
    } else {
        _editData->selectedGene = removeIndex;
    }
}
