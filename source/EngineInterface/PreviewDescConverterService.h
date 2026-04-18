#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/PreviewDesc.h>

struct ConversionResult
{
    PreviewDesc description;
    float frontAngle = 0;
};

class PreviewDescConverterService
{
    MAKE_SINGLETON(PreviewDescConverterService);

public:
    ConversionResult convertToPreviewDesc(
        GenomeDesc const& genome,
        int startGeneIndex,
        Desc&& phenotype) const;
};