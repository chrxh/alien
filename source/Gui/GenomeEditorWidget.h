#pragma once

#include "Definitions.h"

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(CreatureTabEditData const& editData, CreatureTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(CreatureTabEditData const& genome, CreatureTabLayoutData const& layoutData);

    void processHeaderData();
    void processGeneList();
    void processGeneListButtons();

    void onAddGene();
    void onRemoveGene();
    void onMoveGeneUpward();
    void onMoveGeneDownward();

    void removeGeneIntern();
    void moveGeneUpwardIntern();
    void moveGeneDownwardIntern();

    CreatureTabEditData _editData;
    CreatureTabLayoutData _layoutData;
};
