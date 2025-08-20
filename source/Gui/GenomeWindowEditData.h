#pragma once

#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

struct _GenomeWindowEditData
{
    std::optional<int> currentPreviewId;   // TabId of the current preview
    std::unordered_map<GenomeDescriptionWithStartGeneIndex, CollectionDescription> genotypeToPhenotypeCache;
};
