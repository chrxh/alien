#pragma once

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _CreaturePreviewWidget
{
public:
    static CreaturePreviewWidget
    create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDescription const& genomeWithStartIndex);

    void process(CollectionDescription&& phenotype);

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

    void processContent(ConversionResult const& conversionResult);
    void processActionButtons();

    GenomeTabEditData _editData;
    GeneIndicesForSubGenome _geneIndices;
    SubGenomeDescription _subGenome;
    uint64_t _creatureId = 0;
    std::optional<float> _visualFrontAngle;

    float _zoom = 20.0f;
    std::optional<RealVector2D> _windowSizeFromPreviousFrame;
    std::optional<float> _zoomFromPreviousFrame;
    bool _initialScrollPositionSet = false;
};