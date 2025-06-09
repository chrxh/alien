#pragma once

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

    CreatureTabEditData _editData;
    CreatureTabLayoutData _layoutData;
};
