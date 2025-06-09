#include "NodeEditorWidget.h"

#include "AlienImGui.h"
#include "CreatureTabEditData.h"
#include "LoginDialog.h"

NodeEditorWidget _NodeEditorWidget::create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(editData, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        if (getSelectedNode()) {
            processNodeAttributes();
        } else {
            processNoSelection();
        }
    }
    ImGui::EndChild();
}

_NodeEditorWidget::_NodeEditorWidget(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
    : _editData(editData)
    , _layoutData(layoutData)
{
}

void _NodeEditorWidget::processNodeAttributes()
{
    AlienImGui::Group("Selected node");
}

void _NodeEditorWidget::processNoSelection()
{
    AlienImGui::Group("Selected node");
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        AlienImGui::DisabledField();
        auto text = "No node is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}

std::optional<int> _NodeEditorWidget::getSelectedNode() const
{
    if (!_editData->selectedGene.has_value()) {
        return std::nullopt;
    }

    auto geneIndex = _editData->selectedGene.value();
    if (!_editData->selectedNodeByGeneIndex.contains(geneIndex)) {
        return std::nullopt;
    }

    return _editData->selectedNodeByGeneIndex.at(geneIndex);
}
