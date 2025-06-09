#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"

class GenomeDescriptionEditService
{
    MAKE_SINGLETON(GenomeDescriptionEditService);

public:
    void addEmptyGene(GenomeDescription_New& genome, int index);    // Adds empty gene after index
    void removeGene(GenomeDescription_New& genome, int index);
    void swapGenes(GenomeDescription_New& genome, int index);   // Swaps gene at index with gene at index + 1

    void addEmptyNode(GeneDescription& gene, int index);  // Adds empty node after index
};
