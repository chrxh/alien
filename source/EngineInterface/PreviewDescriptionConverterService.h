#pragma once

#include "Base/Singleton.h"

#include "EngineInterface/Description.h"
#include "EngineInterface/PreviewDescriptions.h"

struct ConversionResult
{
    PreviewDescription description;
    float frontAngle = 0;
    std::optional<float> visualFrontAngle;
};

class PreviewDescriptionConverterService
{
    MAKE_SINGLETON(PreviewDescriptionConverterService);

public:
    ConversionResult convertToPreviewDescription(
        GenomeDescription const& genome,
        int startGeneIndex,
        Description&& phenotype,
        std::optional<float> const& lastVisualFrontAngle) const;
};