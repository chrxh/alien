#pragma once

#include "Base/Singleton.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/PreviewDescriptions.h"

class PreviewDescriptionConverterService
{
    MAKE_SINGLETON(PreviewDescriptionConverterService);

public:
    PreviewDescription convert(GenomeDescription const& genome, CollectionDescription&& phenotype, int rootGeneIndex) const;
};