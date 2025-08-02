#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "GenomeDescription.h"
#include "SimulationParameters.h"

class GenomeDescriptionInfoService
{
    MAKE_SINGLETON(GenomeDescriptionInfoService);

public:
    int getNumberOfNodes(GenomeDescription const& genome) const;
    int getNumberOfResultingCells(GenomeDescription const& genome) const;  // Returns -1 for infinite
    std::vector<int> getReferences(GeneDescription const& gene) const;
    std::vector<int> getReferencedBy(GenomeDescription const& genome, int geneIndex) const;
    int getRootGene(GenomeDescription const& genome, int startGeneIndex) const;

private:
    std::set<int> calcIndicesOfGeneHull(GenomeDescription const& genome) const;
};
