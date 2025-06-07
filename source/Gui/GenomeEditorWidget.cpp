#include "GenomeEditorWidget.h"

#include <imgui.h>

#include "EngineInterface/GenomeDescriptionInfoService.h"

#include "AlienImGui.h"
#include "CreatureTabGenomeData.h"
#include "CreatureTabLayoutData.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderTotalWidth = 400.0f;
    auto constexpr HeaderRightColumnWidth = 120.0f;
}

GenomeEditorWidget _GenomeEditorWidget::create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
{
    return GenomeEditorWidget(new _GenomeEditorWidget(genome, layoutData));
}

void _GenomeEditorWidget::process()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_layoutData->genomeEditorWidth, 0))) {
        processHeaderData();

        AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->geneListHeight);

        processGeneList();
    }
    ImGui::EndChild();
}

_GenomeEditorWidget::_GenomeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _genome(genome)
    , _layoutData(layoutData)
{}

void _GenomeEditorWidget::processHeaderData()
{
    AlienImGui::Group("Genome");

    auto availableWidth = scaleInverse(ImGui::GetContentRegionAvail().x);
    auto width = scale(std::min(availableWidth, HeaderTotalWidth));
    if (ImGui::BeginChild("GenomeHeader", ImVec2(width, ImGui::GetContentRegionAvail().y - _layoutData->geneListHeight), 0)) {
        auto numNodesString = std::to_string(GenomeDescriptionInfoService::get().getNumberOfNodes(_genome->genome));
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Nodes").readOnly(true).textWidth(HeaderRightColumnWidth), numNodesString);
        auto numCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(_genome->genome);
        auto numCellsString = numCells != -1 ? std::to_string(numCells) : std::string("infinity");
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Resulting cells").readOnly(true).textWidth(HeaderRightColumnWidth), numCellsString);
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Front angle").format("%.1f").textWidth(HeaderRightColumnWidth), _genome->genome._frontAngle);
    }
    ImGui::EndChild();
}

void _GenomeEditorWidget::processGeneList()
{
    if (ImGui::BeginChild("GeneList", ImVec2(0, 0))) {
        static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_RowBg
            | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

        if (ImGui::BeginTable("Gene list", 3, flags, ImVec2(-1, -1), 0.0f)) {
            ImGui::TableSetupColumn("Gene", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
            ImGui::TableSetupColumn("Nodes", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
            ImGui::TableSetupColumn("Shape", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scale(140.0f));
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
