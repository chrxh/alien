#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "Definitions.h"

struct _CreatureTabEditData
{
    GenomeDescription_New genome;
    std::optional<int> selectedGene;
    std::map<int, int> selectedNodeByGeneIndex;
};
