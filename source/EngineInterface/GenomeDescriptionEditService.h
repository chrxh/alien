#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "GenomeDescription.h"
#include "SimulationParameters.h"

class GenomeDescriptionEditService
{
    MAKE_SINGLETON(GenomeDescriptionEditService);

public:
    void addGene(GenomeDescription& genome, int index, GeneDescription const& newGene);    // Adds gene after index
    void removeGene(GenomeDescription& genome, int index);
    void swapGenes(GenomeDescription& genome, int index);  // Swaps gene at index with gene at index + 1

    void addNode(GeneDescription& gene, int index, NodeDescription const& node);  // Adds node after index
    void removeNode(GeneDescription& gene, int index);
    void swapNodes(GeneDescription& gene, int index);  // Swaps node at index with node at index + 1

    using CreatureGeneIndices = std::vector<int>;
    std::vector<GenomeDescription> createDescriptionsForPreview(GenomeDescription const& genome, std::vector<CreatureGeneIndices> const& creatureGenesVec);

//private:
    void adaptDescriptionForPreview(GenomeDescription& genome, int startGeneIndex);
};
