#pragma once

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _CreaturePreviewWidget
{
public:
    static CreaturePreviewWidget
    create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDescription const& genomeWithStartIndex);

    void process(CollectionDescription&& phenotype, float width);

    uint64_t getCreatureId() const;
    void setCreatureId(uint64_t value);

    GeneIndicesForSubGenome const& getGeneIndices() const;
    void setGeneIndices(GeneIndicesForSubGenome const& value);

    SubGenomeDescription const& getGenomeWithStartIndex() const;
    void setGenomeWithStartIndex(SubGenomeDescription const& value);

    void resetVisualFrontAngle();

private:
    _CreaturePreviewWidget(
        GenomeTabEditData const& editData,
        GeneIndicesForSubGenome const& geneIndices,
        SubGenomeDescription const& genomeWithStartIndex);

    void processCellGraph(ConversionResult const& conversionResult);
    void processActionButtons();
    void processTitle();
    RealVector2D mapWorldToViewPosition(RealVector2D const& worldPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos) const;
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos) const;

    SimulationScrollbars _scrollbars;

    GenomeTabEditData _editData;
    GeneIndicesForSubGenome _geneIndices;
    SubGenomeDescription _subGenome;
    uint64_t _creatureId = 0;
    std::optional<float> _visualFrontAngle;

    RealVector2D _worldCenter;
    float _zoom = 20.0f;
    bool _initialScrollPositionSet = false;
};