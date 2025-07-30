#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _GeneEditorWidget
{
public:
    static GeneEditorWidget create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void process();

private:
    _GeneEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void processNoSelection();
    void processHeaderData();
    void processNodeList();
    void processNodeListButtons();

    void onAddNode();
    void onRemoveNode();
    void onMoveNodeUpward();
    void onMoveNodeDownward();

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    std::optional<int> _selectedNodeFromPreviousFrame;
};
