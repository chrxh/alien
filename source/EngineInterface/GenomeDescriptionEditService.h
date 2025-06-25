#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "CreatureDescription.h"
#include "SimulationParameters.h"

class GenomeDescriptionEditService
{
    MAKE_SINGLETON(GenomeDescriptionEditService);

public:
    void addGene(CreatureDescription& genome, int index, GeneDescription const& newGene);    // Adds gene after index
    void removeGene(CreatureDescription& genome, int index);
    void swapGenes(CreatureDescription& genome, int index);   // Swaps gene at index with gene at index + 1

    void addEmptyNode(GeneDescription& gene, int index);  // Adds empty node after index
    void removeNode(GeneDescription& gene, int index);
    void swapNodes(GeneDescription& gene, int index);  // Swaps node at index with node at index + 1
};
