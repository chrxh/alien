#include "GenomeEditorWidget.h"

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/join.hpp>
#include <imgui.h>
#include <ranges>

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
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Node count").readOnly(true).textWidth(HeaderRightColumnWidth), numNodesString);
        auto numCells = GenomeDescriptionInfoService::get().getNumberOfResultingCells(_genome->genome);
        auto numCellsString = numCells != -1 ? std::to_string(numCells) : std::string("Infinity");
        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Resulting cells").readOnly(true).textWidth(HeaderRightColumnWidth), numCellsString);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Front angle").format("%.1f").textWidth(HeaderRightColumnWidth), _genome->genome._frontAngle);
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

            auto const& genome = _genome->genome;

            ImGuiListClipper clipper;
            clipper.Begin(genome._genes.size());
            while (clipper.Step()) {
                for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                    auto const& gene = genome._genes.at(row);

                    ImGui::PushID(row);
                    ImGui::TableNextRow(0, scale(23.0f));

                    // Column 0: No.
                    ImGui::TableNextColumn();
                    AlienImGui::Text(std::to_string(row + 1));

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
                    auto referencedGenesStrings = referencedGenes 
                        | std::views::transform([](auto const& geneIndex) { return std::to_string(geneIndex + 1); });
                    auto referencedGenesString = boost::algorithm::join(std::vector(referencedGenesStrings.begin(), referencedGenesStrings.end()), ", ");
                    AlienImGui::Text(referencedGenesString);

                    // Column 5: Referencing genes
                    ImGui::TableNextColumn();
                    auto referencingGenes = GenomeDescriptionInfoService::get().getReferencedGeneIndices(gene);
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
