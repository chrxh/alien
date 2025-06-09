#pragma once

#include "Definitions.h"

class _GeneEditorWidget
{
public:
    static GeneEditorWidget create(CreatureTabGenomeData const& editData, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GeneEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void processNoSelection();
    void processHeaderData();
    void processNodeList();

    CreatureTabGenomeData _editData;
    CreatureTabLayoutData _layoutData;
};
