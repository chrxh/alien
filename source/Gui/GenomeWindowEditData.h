#pragma once

#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/GenomeDescriptionEditService.h"

#include "Definitions.h"

struct _GenomeWindowEditData
{
    std::optional<int> currentPreviewId;   // TabId of the current preview
    GenotypeToPhenotypeCache genotypeToPhenotypeCache;
};
