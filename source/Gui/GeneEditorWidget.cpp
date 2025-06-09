#include "GeneEditorWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "CreatureTabGenomeData.h"
#include "CreatureTabLayoutData.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderTotalWidth = 400.0f;
    auto constexpr HeaderRightColumnWidth = 120.0f;
}

GeneEditorWidget _GeneEditorWidget::create(CreatureTabGenomeData const& editData, CreatureTabLayoutData const& layoutData)
{
    return GeneEditorWidget(new _GeneEditorWidget(editData, layoutData));
}

void _GeneEditorWidget::process()
{
    if (ImGui::BeginChild("GeneEditor", ImVec2(_layoutData->geneEditorWidth, 0))) {
        if (_editData->selectedGene.has_value()) {
            processHeaderData();

            AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->nodeListHeight);

            processNodeList();
        } else {
            processNoSelection();
        }
    }
    ImGui::EndChild();
}

_GeneEditorWidget::_GeneEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _editData(genome)
    , _layoutData(layoutData)
{}

void _GeneEditorWidget::processNoSelection()
{
    AlienImGui::Group("Selected gene");
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
            {startPos.x, startPos.y},
            {startPos.x + size.x, startPos.y + size.y},
            Const::DisabledOverlayColor1,
            Const::DisabledOverlayColor2,
            Const::DisabledOverlayColor1,
            Const::DisabledOverlayColor2);
        auto text = "No gene is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::processHeaderData()
{
    AlienImGui::Group("Selected gene");

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
            ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
            ImGui::TableSetupColumn("Angle", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableHeadersRow();
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, Const::TableHeaderColor);

            ImGuiListClipper clipper;
            clipper.Begin(/*size*/ 10);
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    //auto const& entry = _savepointTable->at(row);

                    ImGui::PushID(row);
                    ImGui::TableNextRow(0, scale(23.0f));

                    ImGui::TableNextColumn();
                    ImGui::TableNextColumn();
                    ImGui::TableNextColumn();
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
}
