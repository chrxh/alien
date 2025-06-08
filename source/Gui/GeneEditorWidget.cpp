#include "GeneEditorWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
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
        processHeaderData();

        AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->nodeListHeight);

        processNodeList();
    }
    ImGui::EndChild();
}

_GeneEditorWidget::_GeneEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _editData(genome)
    , _layoutData(layoutData)
{}

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
