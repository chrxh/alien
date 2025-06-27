#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "CreatureDescription.h"
#include "SimulationParameters.h"

class CreatureDescriptionInfoService
{
    MAKE_SINGLETON(CreatureDescriptionInfoService);

public:
    int getNumberOfNodes(CreatureDescription const& genome) const;
    int getNumberOfResultingCells(CreatureDescription const& genome) const;  // Returns -1 for infinite
    std::vector<int> getReferences(GeneDescription const& gene) const;
    std::vector<int> getReferencedBy(CreatureDescription const& genome, int geneIndex) const;
};
