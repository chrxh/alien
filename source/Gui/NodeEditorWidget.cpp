#include "NodeEditorWidget.h"

#include "AlienImGui.h"
#include "LoginDialog.h"

NodeEditorWidget _NodeEditorWidget::create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(genome, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        AlienImGui::Group("Selected node");
    }
    ImGui::EndChild();
}

_NodeEditorWidget::_NodeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _genome(genome)
    , _layoutData(layoutData)
{
}
