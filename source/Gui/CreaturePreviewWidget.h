#pragma once

#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _CreaturePreviewWidget
{
public:
    static CreaturePreviewWidget
    create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex);

    bool process(CollectionDescription&& phenotype);

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

    GenomeTabEditData _editData;
    GeneIndicesForSubGenome _geneIndices;
    GenomeDescriptionWithStartGeneIndex _genomeWithStartIndex;
    uint64_t _creatureId = 0;
    std::optional<float> _visualFrontAngle;

    float _zoom = 20.0f;
    std::optional<RealVector2D> _lastWindowSize;
    bool _initialScrollPositionSet = false;
};