#include "CreatureTabWidget.h"

#include <imgui.h>

#include "AlienImGui.h"
#include "CreatureTabLayoutData.h"
#include "StyleRepository.h"

CreatureTabWidget _CreatureTabWidget::createDraftCreatureTab(GenomeDescription_New const& genome, std::optional<CreatureTabLayoutData> const& creatureTabLayoutData)
{
    return CreatureTabWidget(new _CreatureTabWidget(genome, creatureTabLayoutData));
}

void _CreatureTabWidget::process()
{
    doLayout();

    ImGui::PushID(_id);

    if (ImGui::BeginChild("Editors", ImVec2(0, ImGui::GetContentRegionAvail().y - _creatureTabLayoutData->_previewsHeight), 0)) {
        processEditors();
    }
    ImGui::EndChild();

    AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _creatureTabLayoutData->_previewsHeight);

    if (ImGui::BeginChild("Previews", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        processPreviews();
    }
    ImGui::EndChild();

    ImGui::PopID();
}

std::string _CreatureTabWidget::getName() const
{
    if (std::holds_alternative<DraftCreature>(_creature)) {
        return "Draft " + std::to_string(_id);
    }
    return "";
}

_CreatureTabWidget::_CreatureTabWidget(GenomeDescription_New const& genome, std::optional<CreatureTabLayoutData> const& creatureTabLayoutData)
{
    static int _sequence = 0;
    _id = ++_sequence;

    _creature = DraftCreature{._genome = genome};
    _creatureTabLayoutData = creatureTabLayoutData;
}


void _CreatureTabWidget::processEditors()
{
    if (ImGui::BeginChild("GenomeEditor", ImVec2(_creatureTabLayoutData->_genomeEditorWidth, 0))) {
        processGenomeEditor();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(1);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _creatureTabLayoutData->_genomeEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    if (ImGui::BeginChild("GeneEditor", ImVec2(_creatureTabLayoutData->_geneEditorWidth, 0))) {
        processGeneEditor();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::PushID(2);
    AlienImGui::MovableVerticalSeparator(AlienImGui::MovableVerticalSeparatorParameters().additive(true), _creatureTabLayoutData->_geneEditorWidth);
    ImGui::PopID();

    ImGui::SameLine();
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        processNodeEditor();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processPreviews()
{
    if (ImGui::BeginChild("DesiredConfigurationPreview", ImVec2(_creatureTabLayoutData->_desiredConfigurationPreviewWidth, 0))) {
        processDesiredConfigurationPreview();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    AlienImGui::MovableVerticalSeparator(
        AlienImGui::MovableVerticalSeparatorParameters().additive(true), _creatureTabLayoutData->_desiredConfigurationPreviewWidth);

    ImGui::SameLine();
    if (ImGui::BeginChild("ActualConfigurationPreview", ImVec2(0, 0))) {
        processActualConfigurationPreview();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processGenomeEditor()
{
    if (ImGui::BeginChild("GenomeHeader", ImVec2(0, ImGui::GetContentRegionAvail().y - _creatureTabLayoutData->_geneListHeight), 0)) {
        AlienImGui::Group("Genome");
    }
    ImGui::EndChild();

    AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _creatureTabLayoutData->_geneListHeight);

    if (ImGui::BeginChild("GeneList", ImVec2(0, 0))) {
        //AlienImGui::Group("Genes");
        processGeneList();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processGeneEditor()
{
    if (ImGui::BeginChild("GeneHeader", ImVec2(0, ImGui::GetContentRegionAvail().y - _creatureTabLayoutData->_nodeListHeight), 0)) {
        AlienImGui::Group("Selected gene");
    }
    ImGui::EndChild();

    AlienImGui::MovableHorizontalSeparator(AlienImGui::MovableHorizontalSeparatorParameters().additive(false), _creatureTabLayoutData->_nodeListHeight);

    if (ImGui::BeginChild("NodeList", ImVec2(0, 0))) {
        //AlienImGui::Group("Genes");
        processNodeList();
    }
    ImGui::EndChild();
}

void _CreatureTabWidget::processNodeEditor()
{
    AlienImGui::Group("Selected node");
}

void _CreatureTabWidget::processDesiredConfigurationPreview()
{
    AlienImGui::Group("Preview (predicted)");
}

void _CreatureTabWidget::processActualConfigurationPreview()
{
    AlienImGui::Group("Preview (simulated)");
}

void _CreatureTabWidget::processGeneList()
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
        clipper.Begin(/*size*/10);
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

void _CreatureTabWidget::processNodeList()
{
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

void _CreatureTabWidget::doLayout()
{
    if (_lastGenomeEditorWidth.has_value()) {
        _creatureTabLayoutData->_geneEditorWidth += _lastGenomeEditorWidth.value() - _creatureTabLayoutData->_genomeEditorWidth;
    }

    if (!_creatureTabLayoutData.has_value()) {
        auto width = ImGui::GetContentRegionAvail().x;
        auto height = ImGui::GetContentRegionAvail().y;
        CreatureTabLayoutData creatureTabLayoutData;
        creatureTabLayoutData._genomeEditorWidth = width / 3;
        creatureTabLayoutData._geneEditorWidth = width / 3;
        creatureTabLayoutData._previewsHeight = height / 2;
        creatureTabLayoutData._desiredConfigurationPreviewWidth = width / 2;
        creatureTabLayoutData._geneListHeight = height / 4;
        creatureTabLayoutData._nodeListHeight = height / 4;
        _creatureTabLayoutData = creatureTabLayoutData;
    }

    auto windowSize = ImGui::GetWindowSize();
    if (_lastWindowSize.has_value() && _lastWindowSize->x > 0 && _lastWindowSize->y > 0) {
        if (_lastWindowSize->x != windowSize.x || _lastWindowSize->y != windowSize.y) {
            auto scalingX = windowSize.x / _lastWindowSize->x;
            auto scalingY = windowSize.y / _lastWindowSize->y;
            _creatureTabLayoutData->_genomeEditorWidth *= scalingX;
            _creatureTabLayoutData->_geneEditorWidth *= scalingX;
            _creatureTabLayoutData->_previewsHeight *= scalingY;
            _creatureTabLayoutData->_desiredConfigurationPreviewWidth *= scalingX;
            _creatureTabLayoutData->_geneListHeight *= scalingY;
            _creatureTabLayoutData->_nodeListHeight *= scalingY;
        }
    }
    _lastWindowSize = {windowSize.x, windowSize.y};

    _creatureTabLayoutData->_genomeEditorWidth = std::max(scale(50.0f), _creatureTabLayoutData->_genomeEditorWidth);
    _creatureTabLayoutData->_geneEditorWidth = std::max(scale(50.0f), _creatureTabLayoutData->_geneEditorWidth);
    _creatureTabLayoutData->_desiredConfigurationPreviewWidth = std::max(scale(50.0f), _creatureTabLayoutData->_desiredConfigurationPreviewWidth);
    _creatureTabLayoutData->_previewsHeight =
        std::min(ImGui::GetContentRegionAvail().y - scale(50.0f), std::max(scale(50.0f), _creatureTabLayoutData->_previewsHeight));

    _lastGenomeEditorWidth = _creatureTabLayoutData->_genomeEditorWidth;
}
