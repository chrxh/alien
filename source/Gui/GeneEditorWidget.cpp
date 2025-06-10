#include "GeneEditorWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"

#include "EngineInterface/GenomeDescriptionEditService.h"

#include "AlienGui.h"
#include "CreatureTabEditData.h"
#include "CreatureTabLayoutData.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderTotalWidth = 400.0f;
    auto constexpr HeaderRightColumnWidth = 120.0f;
}

GeneEditorWidget _GeneEditorWidget::create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
{
    return GeneEditorWidget(new _GeneEditorWidget(editData, layoutData));
}

void _GeneEditorWidget::process()
{
    if (ImGui::BeginChild("GeneEditor", ImVec2(_layoutData->geneEditorWidth, 0))) {
        if (_editData->selectedGeneIndex.has_value()) {
            ImGui::PushID(_editData->selectedGeneIndex.value());
            processHeaderData();

            AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->nodeListHeight);

            processNodeList();
            processNodeListButtons();
            ImGui::PopID();
        } else {
            processNoSelection();
        }
    }
    ImGui::EndChild();
}

_GeneEditorWidget::_GeneEditorWidget(CreatureTabEditData const& genome, CreatureTabLayoutData const& layoutData)
    : _editData(genome)
    , _layoutData(layoutData)
{}

void _GeneEditorWidget::processNoSelection()
{
    AlienGui::Group("Selected gene");
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        AlienGui::DisabledField();
        auto text = "No gene is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::processHeaderData()
{
    AlienGui::Group("Selected gene");

    auto availableWidth = scaleInverse(ImGui::GetContentRegionAvail().x);
    auto width = scale(std::min(availableWidth, HeaderTotalWidth));

    if (ImGui::BeginChild("GeneHeader", ImVec2(width, ImGui::GetContentRegionAvail().y - _layoutData->nodeListHeight), 0)) {
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::processNodeList()
{
    if (ImGui::BeginChild("NodeList", ImVec2(0, 0))) {
        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTable("Node list", 3, flags, ImVec2(-1, -1), 0.0f)) {
            ImGui::TableSetupColumn("No.", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(30.0f));
            ImGui::TableSetupColumn("Node type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(80.0f));
            ImGui::TableSetupColumn("Angle", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(40.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

            auto const& gene = _editData->getSelectedGeneRef();

            ImGuiListClipper clipper;
            clipper.Begin(gene._nodes.size());
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    auto const& node = gene._nodes.at(row);

                    ImGui::PushID(row);
                    ImGui::TableNextRow(0, scale(21.0f));

                    // Column 0: No.
                    ImGui::TableNextColumn();
                    AlienGui::Text(std::to_string(row + 1));
                    ImGui::SameLine();
                    auto selectedNode = _editData->getSelectedNodeIndex();
                    auto selected = selectedNode ? selectedNode.value() == row : false;
                    if (ImGui::Selectable(
                            "",
                            &selected,
                            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap,
                            ImVec2(0, scale(ImGui::GetTextLineHeightWithSpacing()) - ImGui::GetStyle().FramePadding.y))) {
                        if (selected) {
                            _editData->setSelectedNodeIndex(row);
                        }
                    }

                    // Column 1: Node type
                    ImGui::TableNextColumn();
                    AlienGui::Text(Const::CellTypeGenomeStrings.at(node.getCellType()));

                    // Column 2: Angle
                    ImGui::TableNextColumn();
                    AlienGui::Text(StringHelper::format(node._referenceAngle, 1));
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::processNodeListButtons()
{
    auto cursorPos = ImGui::GetCursorScreenPos();

    ImVec2 buttonGroupSize = {scale(108.0f), scale(22.0f)};
    ImGui::SetCursorScreenPos(
        ImVec2(cursorPos.x + ImGui::GetContentRegionAvail().x - buttonGroupSize.x - scale(15.0f), cursorPos.y - buttonGroupSize.y - scale(20.0f)));
    if (ImGui::BeginChild("ButtonGroup", buttonGroupSize)) {
        auto selectedNode = _editData->getSelectedNodeIndex();
        auto const& gene = _editData->genome._genes.at(_editData->selectedGeneIndex.value());
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_CIRCLE))) {
            onAddNode();
        }
        ImGui::SameLine();
        AlienGui::PaddingLeft();
        ImGui::BeginDisabled(!selectedNode.has_value());
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_MINUS_CIRCLE))) {
            onRemoveNode();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::PaddingLeft();
        ImGui::BeginDisabled(!selectedNode.has_value() || selectedNode.value() == 0);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_UP))) {
            onMoveNodeUpward();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::PaddingLeft();
        ImGui::BeginDisabled(!selectedNode.has_value() || selectedNode.value() == gene._nodes.size() - 1);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_DOWN))) {
            onMoveNodeDownward();
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::onAddNode()
{
    auto& gene = _editData->getSelectedGeneRef();
    auto selectedNode = _editData->getSelectedNodeIndex();
    if (gene._nodes.empty()) {
        GenomeDescriptionEditService::get().addEmptyNode(gene, 0);
        _editData->setSelectedNodeIndex(0);
    } else {
        int insertIndex;
        if (selectedNode.has_value()) {
            insertIndex = selectedNode.value();
        } else {
            insertIndex = toInt(gene._nodes.size()) - 1;
        }

        GenomeDescriptionEditService::get().addEmptyNode(gene, insertIndex);

        _editData->setSelectedNodeIndex(insertIndex + 1);
    }
}

void _GeneEditorWidget::onRemoveNode()
{
    int removeIndex = _editData->getSelectedNodeIndex().value();
    auto& gene = _editData->getSelectedGeneRef();

    GenomeDescriptionEditService::get().removeNode(gene, removeIndex);

    // Adapt node selection
    auto& nodes = gene._nodes;
    if (nodes.empty()) {
        _editData->setSelectedNodeIndex(std::nullopt);
    } else if (removeIndex >= toInt(nodes.size())) {
        _editData->setSelectedNodeIndex(toInt(nodes.size()) - 1);
    } else {
        _editData->setSelectedNodeIndex(removeIndex);
    }
}

void _GeneEditorWidget::onMoveNodeUpward()
{
    auto indexToMove = _editData->getSelectedNodeIndex().value();
    auto& gene = _editData->getSelectedGeneRef();
    GenomeDescriptionEditService::get().swapNodes(gene, indexToMove - 1);

    // Adapt gene selection
    _editData->setSelectedNodeIndex(_editData->getSelectedNodeIndex().value() - 1);
}

void _GeneEditorWidget::onMoveNodeDownward()
{
    auto indexToMove = _editData->getSelectedNodeIndex().value();
    auto& gene = _editData->getSelectedGeneRef();
    GenomeDescriptionEditService::get().swapNodes(gene, indexToMove);

    // Adapt gene selection
    _editData->setSelectedNodeIndex(_editData->getSelectedNodeIndex().value() + 1);
}
