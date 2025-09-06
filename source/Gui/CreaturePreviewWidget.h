#pragma once

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _CreaturePreviewWidget
{
public:
    static CreaturePreviewWidget
    create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex);

    void process(CollectionDescription&& phenotype);

    uint64_t getCreatureId() const;
    void setCreatureId(uint64_t value);

    GeneIndicesForSubGenome const& getGeneIndices() const;
    void setGeneIndices(GeneIndicesForSubGenome const& value);

    GenomeDescriptionWithStartGeneIndex const& getGenomeWithStartIndex() const;
    void setGenomeWithStartIndex(GenomeDescriptionWithStartGeneIndex const& value);

    void resetVisualFrontAngle();

private:
    _CreaturePreviewWidget(
        GenomeTabEditData const& editData,
        GeneIndicesForSubGenome const& geneIndices,
        GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex);

    void processContent(ConversionResult const& conversionResult);
    void processActionButtons();

    struct Dimensions
    {
        RealVector2D upperLeft;
        RealVector2D lowerRight;
    };
    Dimensions calcDimensions(PreviewDescription const& desc) const;

    GenomeTabEditData _editData;
    GeneIndicesForSubGenome _geneIndices;
    GenomeDescriptionWithStartGeneIndex _genomeWithStartIndex;
    uint64_t _creatureId = 0;
    std::optional<float> _visualFrontAngle;

    float _zoom = 20.0f;
    std::optional<RealVector2D> _windowSizeFromPreviousFrame;
    std::optional<float> _zoomFromPreviousFrame;
    bool _initialScrollPositionSet = false;
};