#pragma once

#include "Base/Cache.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

struct _GenomeWindowEditData
{
    std::optional<int> currentPreviewId;   // TabId of the current preview
    Cache<SubGenomeDescription, Description, 100000> genotypeToPhenotypeCache;
};
