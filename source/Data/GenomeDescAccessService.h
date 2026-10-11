#pragma once

#include <optional>
#include <set>
#include <vector>

#include <Base/Singleton.h>

#include "GenomeDesc.h"
#include "SimulationParameters.h"

class GenomeDescAccessService
{
    MAKE_SINGLETON(GenomeDescAccessService);

public:
    int getNumberOfNodes(GenomeDesc const& genome) const;
    int getNumberOfResultingCells(GenomeDesc const& genome, int startGeneIndex = 0) const;  // Returns -1 for infinite
    std::vector<int> getReferences(GeneDesc const& gene) const;
    std::vector<int> getReferencedBy(GenomeDesc const& genome, int geneIndex) const;
    bool isConnectedToRoot(GenomeDesc const& genome, int startGeneIndex) const;
    std::set<int> getReferencedGenesInRootGeneHull(GenomeDesc const& genome) const;
    std::set<int> getReachableGenes(GenomeDesc const& genome, int startGeneIndex) const;

    struct GeneConstruction
    {
        int constructingGeneIndex = 0;
        int constructedGeneIndex = 0;
        std::vector<int> constructorNodeIndices;
        bool anyWithSeparation = false;
    };
    std::vector<GeneConstruction> getGeneConstructions(GenomeDesc const& genome) const;
    std::vector<std::optional<int>> calcDistancesFromRootGene(GenomeDesc const& genome) const;

    using GeneIndicesForSubGenome = std::vector<int>;
    std::vector<GeneIndicesForSubGenome> getGeneIndicesForSubGenomes(GenomeDesc const& genome) const;

private:
    struct ReferencedGenes
    {
        std::vector<int> nonSeparatingGeneIndices;
        std::vector<int> separatingGeneIndices;
    };
    ReferencedGenes getReferencedGenesInNonSeparatingGeneHull(GenomeDesc const& genome, int startGeneIndex) const;
    std::vector<GeneIndicesForSubGenome> dropContainedSubGenomes(GenomeDesc const& genome, std::vector<GeneIndicesForSubGenome> const& subGenomes) const;
};
