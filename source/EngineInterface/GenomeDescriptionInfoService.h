#pragma once

#include <vector>

#include "Base/Singleton.h"

#include "GenomeDescriptions.h"
#include "SimulationParameters.h"

class GenomeDescriptionInfoService
{
    MAKE_SINGLETON(GenomeDescriptionInfoService);

public:
    int getNumberOfNodes(GenomeDescription_New const& genome) const;
    int getNumberOfResultingCells(GenomeDescription_New const& genome) const;  // Returns -1 for infinite
};
