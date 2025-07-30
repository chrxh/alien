#pragma once

#include "Definitions.h"

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(GenomeTabEditData const& genome, GenomeTabLayoutData const& layoutData);

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

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    std::optional<int> _selectedGeneFromPreviousFrame;
};
