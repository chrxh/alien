#pragma once

#include "Definitions.h"

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void processHeaderData();
    void processGeneList();

    CreatureTabGenomeData _genome;
    CreatureTabLayoutData _layoutData;
};
