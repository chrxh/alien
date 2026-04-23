#pragma once

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/GenomeDescEditService.h>

#include "Definitions.h"

struct _GenomeWindowEditData
{
    std::optional<int> currentPreviewId;  // TabId of the current preview
    GenotypeToPhenotypeCache genotypeToPhenotypeCache;
    bool showNodeIndex = true;  // true = show node index, false = show cell function
};
