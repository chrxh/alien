#include "GeneEditorWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"

#include "EngineInterface/GenomeDescriptionEditService.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
}


GeneEditorWidget _GeneEditorWidget::create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
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

_GeneEditorWidget::_GeneEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
    : _editData(editData)
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
    AlienGui::Group("Selected gene", std::nullopt, true);

    auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));
    if (ImGui::BeginChild("GeneHeader", ImVec2(0, -_layoutData->nodeListHeight), 0, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto& gene = _editData->getSelectedGeneRef();

        _editData->updateGeometry(gene._shape); // Do it every time in order to avoid check for changes

        // Gene name
        AlienGui::InputText(AlienGui::InputTextParameters().name("Gene name").textWidth(rightColumnWidth), gene._name);

        // Shape
        AlienGui::Combo(AlienGui::ComboParameters().name("Shape generator").values(Const::ConstructorShapeStrings).textWidth(rightColumnWidth), gene._shape);

        // Angle alignment
        AlienGui::BeginIndent();
        if (gene._shape == ConstructorShape_Custom) {
            AlienGui::Combo(
                AlienGui::ComboParameters().name("Angle alignment").values(Const::ConstructorAlignmentStrings).textWidth(rightColumnWidth),
                gene._angleAlignment);
        } else {
            std::string text = Const::ConstructorAlignmentStrings.at(gene._angleAlignment);
            AlienGui::InputText(AlienGui::InputTextParameters().name("Angle alignment").textWidth(rightColumnWidth).readOnly(true), text);
        }
        AlienGui::EndIndent();

        // Separation
        AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Separation").textWidth(rightColumnWidth), gene._separation);

        // Number of branches
        AlienGui::BeginIndent();
        if (!gene._separation) {
            auto numBranches = gene._numBranches - 1;   // Convert to 0-based for UI (1 branch -> index 0, 2 branches -> index 1, etc.)
            AlienGui::Combo(
                AlienGui::ComboParameters()
                    .name("Number of branches")
                    .values({"1", "2", "3", "4", "5", "6"})
                    .textWidth(rightColumnWidth),
                numBranches);
            gene._numBranches = numBranches + 1;  // Convert back to 1-based (index 0 -> 1 branch, index 1 -> 2 branches, etc.)
        } else {
            std::string text = "-";
            AlienGui::InputText(AlienGui::InputTextParameters().name("Number of branches").textWidth(rightColumnWidth).readOnly(true), text);
        }
        AlienGui::EndIndent();

        // Concatenations
        AlienGui::InputInt(AlienGui::InputIntParameters().name("Concatenations").infinity(true).textWidth(rightColumnWidth), gene._numConcatenations);

        // Connection distance
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Connection distance").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
            gene._connectionDistance);

        // Stiffness
        AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Stiffness").format("%.2f").step(0.05f).textWidth(rightColumnWidth), gene._stiffness);
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::processNodeList()
{
    if (ImGui::BeginChild("NodeList", ImVec2(0, 0))) {
        auto scrollToNodeIndex = -1;
        auto selectedNode = _editData->getSelectedNodeIndex();
        if (!_selectedNodeFromPreviousFrame.has_value() || _selectedNodeFromPreviousFrame != selectedNode) {
            if (selectedNode.has_value()) {
                scrollToNodeIndex = std::max(1, selectedNode.value());
            }
        }
        _selectedNodeFromPreviousFrame = selectedNode;

        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTable("Node list", 5, flags, ImVec2(-1, -1), 0.0f)) {
            ImGui::TableSetupColumn("No.", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(30.0f));
            ImGui::TableSetupColumn("Node type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(135.0f));
            ImGui::TableSetupColumn("Angle", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(40.0f));
            ImGui::TableSetupColumn("Color", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(40.0f));
            ImGui::TableSetupColumn("Signal restriction", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(120.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

            auto const& gene = _editData->getSelectedGeneRef();

            ImGuiListClipper clipper;
            clipper.Begin(gene._nodes.size());
            if (scrollToNodeIndex != -1) {
                clipper.IncludeItemByIndex(scrollToNodeIndex);
            }
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    auto const& node = gene._nodes.at(row);
                    if (row == scrollToNodeIndex) {
                        ImGui::SetScrollHereY();
                    }
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
                    {
                        auto nodeType = node.getCellType();
                        auto text = Const::CellTypeGenomeStrings.at(nodeType);
                        if (nodeType == CellTypeGenome_Constructor) {
                            auto const& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                            text += " (Gene " + std::to_string(constructor._geneIndex + 1) + ")";
                        }
                        AlienGui::Text(text);
                    }

                    // Column 2: Angle
                    ImGui::TableNextColumn();
                    AlienGui::Text(StringHelper::format(node._referenceAngle, 1));

                    // Column 3: Color
                    ImGui::TableNextColumn();
                    if (ImGui::BeginChild("color", {0, ImGui::GetTextLineHeight()}, 0, ImGuiWindowFlags_NoInputs)) {
                        AlienGui::ColorField(Const::IndividualCellColors[node._color], 40.0f, ImGui::GetTextLineHeight());
                    }
                    ImGui::EndChild();

                    // Column 4: Signal restriction
                    ImGui::TableNextColumn();
                    if (node._signalRestriction._active) {
                        if (ImGui::BeginChild("signal", {0, scale(17.0f)}, 0, ImGuiWindowFlags_NoInputs)) {
                            AlienGui::Text(
                                StringHelper::format(node._signalRestriction._baseAngle, 1) + " deg, "
                                + StringHelper::format(node._signalRestriction._openingAngle, 1) + " deg");
                        }
                        ImGui::EndChild();
                    }

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
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_PLUS_CIRCLE).frame(true).transparentBackground(false))) {
            onAddNode();
        }
        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!selectedNode.has_value());
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_MINUS_CIRCLE).frame(true).transparentBackground(false))) {
            onRemoveNode();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!selectedNode.has_value() || selectedNode.value() == 0);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_UP).frame(true).transparentBackground(false))) {
            onMoveNodeUpward();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        AlienGui::MoveTickLeft();
        ImGui::BeginDisabled(!selectedNode.has_value() || selectedNode.value() == gene._nodes.size() - 1);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_CHEVRON_CIRCLE_DOWN).frame(true).transparentBackground(false))) {
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
        GenomeDescriptionEditService::get().addNode(gene, 0, NodeDescription());
        _editData->setSelectedNodeIndex(0);
    } else {
        int insertIndex;
        if (selectedNode.has_value()) {
            insertIndex = selectedNode.value();
        } else {
            insertIndex = toInt(gene._nodes.size()) - 1;
        }
        int color = gene._nodes.at(insertIndex)._color;

        GenomeDescriptionEditService::get().addNode(gene, insertIndex, NodeDescription().color(color));

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
