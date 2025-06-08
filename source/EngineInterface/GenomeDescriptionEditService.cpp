#include "GenomeDescriptionEditService.h"

void GenomeDescriptionEditService::addEmptyGene(GenomeDescription_New& genome, int index)
{
    genome._genes.insert(genome._genes.begin() + index, GeneDescription());
}

void GenomeDescriptionEditService::removeGene(GenomeDescription_New& genome, int index)
{
    genome._genes.erase(genome._genes.begin() + index);
}
