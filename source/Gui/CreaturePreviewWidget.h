#pragma once

#include <EngineInterface/GenomeDescEditService.h>
#include <EngineInterface/PreviewDesc.h>

#include "Definitions.h"
#include "PreviewDescView.h"

class _CreaturePreviewWidget
{
public:
    static CreaturePreviewWidget
    create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDesc const& genomeWithStartIndex);

    void process(bool& phenotypeChanged, ContentDesc& phenotype, GenomeDesc const& genome, float height);

    uint64_t getCreatureId() const;
    void setCreatureId(uint64_t value);

    GeneIndicesForSubGenome const& getGeneIndices() const;
    void setGeneIndices(GeneIndicesForSubGenome const& value);

    SubGenomeDesc const& getGenomeWithStartIndex() const;
    void setGenomeWithStartIndex(SubGenomeDesc const& value);

    PreviewDesc const& getPreviewDesc() const;

    void resetVisualFrontAngle();

private:
    _CreaturePreviewWidget(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDesc const& genomeWithStartIndex);

    void updateViewport();
    void processMouseNavigation();
    void updateSelection();
    void processPreviewRendering();
    void processNeuralActivityEditor(bool& phenotypeChanged, ContentDesc& phenotype);
    void processActionButtons();
    void processScrollbars();
    void processTitle();

    void moveCenter(RealVector2D const& startWorldPosition, RealVector2D const& endViewPos);

    void updatePhenotype(ContentDesc& phenotype, CellPreviewDesc const& editedCell) const;

    SimulationScrollbars _scrollbars;
    PreviewDescView _previewView;

    GenomeTabEditData _editData;
    GeneIndicesForSubGenome _geneIndices;
    SubGenomeDesc _subGenome;
    PreviewDesc _previewDesc;
    uint64_t _creatureId = 0;
    std::optional<float> _visualFrontAngle;
    std::optional<uint64_t> _selectedCellIdFromPreview;

    PreviewViewport _viewport;
    bool _initialScrollPositionSet = false;
    std::optional<int> _selectedNodeFromPreview;

    std::optional<RealVector2D> _worldPosForPanning;
};
