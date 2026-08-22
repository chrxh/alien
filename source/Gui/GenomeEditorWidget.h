#pragma once

#include <EngineInterface/Colors.h>
#include <EngineInterface/GenomeDesc.h>

#include "Definitions.h"
#include "MutationRatesWidget.h"

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void processHeaderData();

    // Genes and their nodes form one hierarchy and are therefore shown as one tree
    void processStructureTree();
    void
    processGeneNode(int geneIndex, GeneDesc const& gene, bool isUnreachable, bool scrollToSelection, ColorVector<FloatColorRGB> const& customizationColors);
    void processNodeLeaf(int geneIndex, int nodeIndex, GeneDesc const& gene, NodeDesc const& node, ColorVector<FloatColorRGB> const& customizationColors);
    void processStructureButtons();

    void onAddGene();
    void onRemoveGene();
    void onMoveGeneUpward();
    void onMoveGeneDownward();

    void onAddNode();
    void onRemoveNode();
    void onMoveNodeUpward();
    void onMoveNodeDownward();

    void removeGeneIntern();
    void moveGeneUpwardIntern();
    void moveGeneDownwardIntern();

    MutationRatesWidget _mutationRatesWidget;

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;
    int _sequenceNumberForCreatedGenes = 0;

    std::optional<int> _selectedGeneFromPreviousFrame;
    bool _selectionChangedFromTree = false;
};
