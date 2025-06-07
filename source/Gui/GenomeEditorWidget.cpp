#include "GenomeEditorWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "CreatureTabLayoutData.h"
#include "StyleRepository.h"

GenomeEditorWidget _GenomeEditorWidget::create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
{
    return GenomeEditorWidget(new _GenomeEditorWidget(genome, layoutData));
}

void _GenomeEditorWidget::process()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_layoutData->genomeEditorWidth, 0))) {
        if (ImGui::BeginChild("GenomeHeader", ImVec2(0, ImGui::GetContentRegionAvail().y - _layoutData->geneListHeight), 0)) {
            AlienImGui::Group("Genome");
        }
        ImGui::EndChild();

        AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->geneListHeight);

        if (ImGui::BeginChild("GeneList", ImVec2(0, 0))) {
            processGeneList();
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
}

_GenomeEditorWidget::_GenomeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _genome(genome)
    , _layoutData(layoutData)
{}

void _GenomeEditorWidget::processGeneList()
{
    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("Gene list", 3, flags, ImVec2(-1, -1), 0.0f)) {
        ImGui::TableSetupColumn("Gene", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Nodes", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
        ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(100.0f));
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
