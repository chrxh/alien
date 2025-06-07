#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "Definitions.h"

struct _CreatureTabGenomeData
{
    GenomeDescription_New genome;
    std::optional<int> selectedGene;
};
