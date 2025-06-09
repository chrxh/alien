#include "GeneEditorWidget.h"

#include <imgui.h>

#include "Base/StringHelper.h"

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
        if (_editData->selectedGene.has_value()) {
            processHeaderData();

            AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->nodeListHeight);

            processNodeList();
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
            ImGui::TableSetupColumn("No.", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
            ImGui::TableSetupColumn("Node type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
            ImGui::TableSetupColumn("Angle", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

            auto const& genome = _editData->genome;
            auto const& gene = genome._genes.at(_editData->selectedGene.value());

            ImGuiListClipper clipper;
            clipper.Begin(gene._nodes.size());
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    auto const& node = gene._nodes.at(row);

                    ImGui::PushID(row);
                    ImGui::TableNextRow(0, scale(23.0f));

                    // Column 0: No.
                    ImGui::TableNextColumn();
                    AlienGui::Text(std::to_string(row + 1));

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
