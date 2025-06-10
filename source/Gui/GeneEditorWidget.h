#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _GeneEditorWidget
{
public:
    static GeneEditorWidget create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GeneEditorWidget(CreatureTabEditData const& genome, CreatureTabLayoutData const& layoutData);

    void processNoSelection();
    void processHeaderData();
    void processNodeList();
    void processNodeListButtons();

    void onAddNode();
    void onRemoveNode();
    void onMoveNodeUpward();
    void onMoveNodeDownward();

    CreatureTabEditData _editData;
    CreatureTabLayoutData _layoutData;
};
