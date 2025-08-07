#pragma once

#include <vector>
#include <set>

#include "Base/Singleton.h"

#include "GenomeDescription.h"
#include "SimulationParameters.h"

struct ReferencedGenesInNonSeparatingGeneHull 
{
    std::vector<int> nonSeparatingGeneIndices;
    std::vector<int> separatingGeneIndices;
};

class GenomeDescriptionInfoService
{
    MAKE_SINGLETON(GenomeDescriptionInfoService);

public:
    int getNumberOfNodes(GenomeDescription const& genome) const;
    int getNumberOfResultingCells(GenomeDescription const& genome) const;  // Returns -1 for infinite
    std::vector<int> getReferences(GeneDescription const& gene) const;
    std::vector<int> getReferencedBy(GenomeDescription const& genome, int geneIndex) const;
    bool isConnectedToRoot(GenomeDescription const& genome, int startGeneIndex) const;
    std::set<int> getReferencedGenesInRootGeneHull(GenomeDescription const& genome) const;
    ReferencedGenesInNonSeparatingGeneHull getReferencedGenesInNonSeparatingGeneHull(GenomeDescription const& genome, int startGeneIndex) const;
};
