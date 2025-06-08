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
};
