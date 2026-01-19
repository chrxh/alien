#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <Base/Cache.h>
#include <Base/Singleton.h>

#include "Description.h"
#include "GenomeDescription.h"
#include "SimulationParameters.h"

using GeneIndicesForSubGenome = std::vector<int>;
using GenotypeToPhenotypeCache = Cache<SubGenomeDesc, Desc, 100000>;

class GenomeDescEditService
{
    MAKE_SINGLETON(GenomeDescEditService);

public:
    void addGene(GenomeDesc& genome, int index, GeneDesc const& newGene) const;  // Adds gene after index
    void removeGene(GenomeDesc& genome, int index) const;
    void swapGenes(GenomeDesc& genome, int index) const;  // Swaps gene at index with gene at index + 1

    void addNode(GeneDesc& gene, int index, NodeDesc const& node) const;  // Adds node after index
    void removeNode(GeneDesc& gene, int index) const;
    void swapNodes(GeneDesc& gene, int index) const;  // Swaps node at index with node at index + 1

    std::vector<SubGenomeDesc> createSubGenomesForPreview(
        GenomeDesc const& genome,
        std::vector<GeneIndicesForSubGenome> const& geneIndicesForSubGenomes,
        bool detailSimulation) const;

    struct SeedCollectionResult
    {
        Desc description;
        std::vector<uint64_t> seedCreatureIds;
    };

    SeedCollectionResult createSeedCollectionForPreview(
        std::vector<SubGenomeDesc> const& subGenomes,
        std::optional<std::reference_wrapper<GenotypeToPhenotypeCache const>> cache = std::nullopt) const;

    std::vector<Desc> extractPhenotypesFromPreview(Desc&& preview, std::vector<uint64_t> const& seedCreatureIds) const;
    void removeSeedFromPhenotype(Desc& phenotype) const;

private:
    Desc createSeedForPreview(SubGenomeDesc const& subGenome, RealVector2D const& pos) const;

    void adaptDescriptionForPreview(GenomeDesc& genome, GeneIndicesForSubGenome const& geneIndices, bool detailSimulation) const;
};
