#pragma once

#include <set>
#include <vector>

#include <Base/Singleton.h>

#include "GenomeDesc.h"
#include "SimulationParameters.h"

class GenomeDescInfoService
{
    MAKE_SINGLETON(GenomeDescInfoService);

public:
    int getNumberOfNodes(GenomeDesc const& genome) const;
    int getNumberOfResultingCells(GenomeDesc const& genome, int startGeneIndex = 0) const;  // Returns -1 for infinite
    std::vector<int> getReferences(GeneDesc const& gene) const;
    std::vector<int> getReferencedBy(GenomeDesc const& genome, int geneIndex) const;
    bool isConnectedToRoot(GenomeDesc const& genome, int startGeneIndex) const;
    std::set<int> getReferencedGenesInRootGeneHull(GenomeDesc const& genome) const;

    using GeneIndicesForSubGenome = std::vector<int>;
    std::vector<GeneIndicesForSubGenome> getGeneIndicesForSubGenomes(GenomeDesc const& genome) const;

private:
    struct ReferencedGenes
    {
        std::vector<int> nonSeparatingGeneIndices;
        std::vector<int> separatingGeneIndices;
    };
    ReferencedGenes getReferencedGenesInNonSeparatingGeneHull(GenomeDesc const& genome, int startGeneIndex) const;
};
