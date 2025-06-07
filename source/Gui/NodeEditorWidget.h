#pragma once

#include "Definitions.h"

class _NodeEditorWidget
{
public:
    static NodeEditorWidget create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _NodeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    CreatureTabGenomeData _genome;
    CreatureTabLayoutData _layoutData;
};
