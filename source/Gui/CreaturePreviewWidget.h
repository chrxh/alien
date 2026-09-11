#pragma once

#include <EngineInterface/GenomeDescEditService.h>
#include <EngineInterface/PreviewDesc.h>

#include "Definitions.h"
#include "PreviewDescRenderer.h"

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

    // Preview of the last processed frame
    PreviewDesc const& getPreviewDesc() const;

    void resetVisualFrontAngle();

private:
    _CreaturePreviewWidget(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDesc const& genomeWithStartIndex);

    void processMouseNavigation();
    void processCellGraphAndSelection();
    void processNeuralActivityEditor(bool& phenotypeChanged, ContentDesc& phenotype);
    void processActionButtons();
    void processScrollbars();
    void processTitle();

    void updateSelection(RealVector2D const& viewSize, RealVector2D const& viewStartPos);

    PreviewViewport createViewport(RealVector2D const& viewSize, RealVector2D const& viewStartPos) const;
    RealVector2D mapWorldToViewPosition(RealVector2D const& worldPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos) const;
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos) const;
    void moveCenter(RealVector2D const& startWorldPosition, RealVector2D const& endViewPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos);

    void updatePhenotype(ContentDesc& phenotype, CellPreviewDesc const& editedCell) const;

    SimulationScrollbars _scrollbars;
    PreviewDescRenderer _renderer;

    GenomeTabEditData _editData;
    GeneIndicesForSubGenome _geneIndices;
    SubGenomeDesc _subGenome;
    PreviewDesc _previewDesc;
    uint64_t _creatureId = 0;
    std::optional<float> _visualFrontAngle;
    std::optional<uint64_t> _selectedCellIdFromPreview;

    RealVector2D _worldCenter;
    float _zoom = 20.0f;
    bool _initialScrollPositionSet = false;
    std::optional<int> _selectedNodeFromPreview;

    std::optional<RealVector2D> _worldPosForPanning;
};
