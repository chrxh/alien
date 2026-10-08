#pragma once

#include <Data/Colors.h>
#include <Data/GenomeDesc.h>

#include "Definitions.h"
#include "GeneGraphWidget.h"
#include "MutationRatesWidget.h"

struct GenomeIssue;

class _GenomeEditorWidget
{
public:
    static GenomeEditorWidget create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void process();

private:
    _GenomeEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData);

    void processHeaderData();

    void processStructureTree();
    void processGeneNode(
        int geneIndex,
        GeneDesc const& gene,
        std::vector<GenomeIssue> const& geneIssues,
        bool scrollToSelection,
        ColorVector<FloatColorRGB> const& customizationColors);
    void processNodeLeaf(
        int geneIndex,
        int nodeIndex,
        GeneDesc const& gene,
        NodeDesc const& node,
        std::vector<GenomeIssue> const& geneIssues,
        ColorVector<FloatColorRGB> const& customizationColors);
    void processStructureButtons();
    void processValidation();

    void onAddGene();
    void onRemoveGene();
    void onMoveGeneUpward();
    void onMoveGeneDownward();

    void onAddNode();
    void onRemoveNode();
    void onMoveNodeUpward();
    void onMoveNodeDownward();

    void onFixAllIssues();

    void removeGeneIntern();
    void moveGeneUpwardIntern();
    void moveGeneDownwardIntern();
    void fixAllIssuesIntern();

    MutationRatesWidget _mutationRatesWidget;
    GeneGraphWidget _geneGraphWidget;
    float _validationHeight = 0;

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;
    int _sequenceNumberForCreatedGenes = 0;

    std::optional<int> _selectedGeneFromPreviousFrame;
    bool _selectionChangedFromTree = false;
};
