#include "NodeEditorWidget.h"

#include "LoginDialog.h"

NodeEditorWidget _NodeEditorWidget::create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(genome, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
    }
    ImGui::EndChild();
}

_NodeEditorWidget::_NodeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData)
    : _genome(genome)
    , _layoutData(layoutData)
{
}
