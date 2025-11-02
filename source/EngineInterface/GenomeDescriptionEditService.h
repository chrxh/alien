#pragma once

#include <vector>

#include <optional>
#include <functional>

#include "Base/Cache.h"
#include "Base/Singleton.h"

#include "Description.h"
#include "GenomeDescription.h"
#include "SimulationParameters.h"

using GeneIndicesForSubGenome = std::vector<int>;
using GenotypeToPhenotypeCache = Cache<SubGenomeDescription, Description, 100000>;

class GenomeDescriptionEditService
{
    MAKE_SINGLETON(GenomeDescriptionEditService);

public:
    void addGene(GenomeDescription& genome, int index, GeneDescription const& newGene) const;  // Adds gene after index
    void removeGene(GenomeDescription& genome, int index) const;
    void swapGenes(GenomeDescription& genome, int index) const;  // Swaps gene at index with gene at index + 1

    void addNode(GeneDescription& gene, int index, NodeDescription const& node) const;  // Adds node after index
    void removeNode(GeneDescription& gene, int index) const;
    void swapNodes(GeneDescription& gene, int index) const;  // Swaps node at index with node at index + 1

    std::vector<SubGenomeDescription> createSubGenomesForPreview(
        GenomeDescription const& genome,
        std::vector<GeneIndicesForSubGenome> const& geneIndicesForSubGenomes,
        bool detailSimulation) const;

    struct SeedCollectionResult
    {
        Description description; 
        std::vector<uint64_t> seedCreatureIds;
    };
    
    SeedCollectionResult createSeedCollectionForPreview(
        std::vector<SubGenomeDescription> const& subGenomes,
        std::optional<std::reference_wrapper<GenotypeToPhenotypeCache const>> cache = std::nullopt) const;
        
    std::vector<Description> extractPhenotypesFromPreview(Description&& preview, std::vector<uint64_t> const& seedCreatureIds) const;
    void removeSeedFromPhenotype(Description& phenotype) const;

private:
    Description createSeedForPreview(SubGenomeDescription const& subGenome, RealVector2D const& pos) const;

    void adaptDescriptionForPreview(GenomeDescription& genome, GeneIndicesForSubGenome const& geneIndices, bool detailSimulation) const;
};
