#pragma once

#include "Definitions.h"

class _GeneEditorWidget
{
public:
    static GeneEditorWidget create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GeneEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void processHeaderData();
    void processNodeList();

    CreatureTabGenomeData _genome;
    CreatureTabLayoutData _layoutData;
};
