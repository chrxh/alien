#include "NodeEditorWidget.h"

#include "AlienGui.h"
#include "CreatureTabEditData.h"
#include "LoginDialog.h"

NodeEditorWidget _NodeEditorWidget::create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(editData, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        if (_editData->getSelectedNode()) {
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
    AlienGui::Group("Selected node");
}

void _NodeEditorWidget::processNoSelection()
{
    AlienGui::Group("Selected node");
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        AlienGui::DisabledField();
        auto text = "No node is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}
