#pragma once

#include "Definitions.h"

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(CreatureTabGenomeData const& editData, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(CreatureTabGenomeData const& genome, CreatureTabLayoutData const& layoutData);

    void processHeaderData();
    void processGeneList();
    void processGeneListButtons();

    void onAddGene();
    void onRemoveGene();

    void removeGeneIntern();

    CreatureTabGenomeData _editData;
    CreatureTabLayoutData _layoutData;
};
