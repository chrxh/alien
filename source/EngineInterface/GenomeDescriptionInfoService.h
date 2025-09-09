#pragma once

#include <vector>
#include <set>

#include "Base/Singleton.h"

#include "GenomeDescription.h"
#include "SimulationParameters.h"

class GenomeDescriptionInfoService
{
    MAKE_SINGLETON(GenomeDescriptionInfoService);

public:
    int getNumberOfNodes(GenomeDescription const& genome) const;
    int getNumberOfResultingCells(GenomeDescription const& genome, int startGeneIndex = 0) const;  // Returns -1 for infinite
    std::vector<int> getReferences(GeneDescription const& gene) const;
    std::vector<int> getReferencedBy(GenomeDescription const& genome, int geneIndex) const;
    bool isConnectedToRoot(GenomeDescription const& genome, int startGeneIndex) const;
    std::set<int> getReferencedGenesInRootGeneHull(GenomeDescription const& genome) const;

    using GeneIndicesForSubGenome = std::vector<int>;
    std::vector<GeneIndicesForSubGenome> getGeneIndicesForSubGenomes(GenomeDescription const& genome) const;

private:
    std::vector<GeneIndicesForSubGenome> getGeneIndicesForSubGenomes(GenomeDescription const& genome, std::set<int> const& nonInspectedGeneIndices, int startGeneIndex) const;

    struct ReferencedGenes
    {
        std::vector<int> nonSeparatingGeneIndices;
        std::vector<int> separatingGeneIndices;
    };
    ReferencedGenes getReferencedGenesInNonSeparatingGeneHull(GenomeDescription const& genome, int startGeneIndex) const;
};
