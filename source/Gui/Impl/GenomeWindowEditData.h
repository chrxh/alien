#pragma once

#include <Data/GenomeDesc.h>
#include <Data/GenomeDescEditService.h>

#include "Definitions.h"

struct _GenomeWindowEditData
{
    std::optional<int> currentPreviewId;  // TabId of the current preview
    GenotypeToPhenotypeCache genotypeToPhenotypeCache;
    bool defaultShowNodeIndex = true;  // true = show node index, false = show cell function
    bool defaultDetailSimulation = false;
    bool defaultShowNeuralActivityEditor = true;
};
