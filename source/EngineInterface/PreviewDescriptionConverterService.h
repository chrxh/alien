#pragma once

#include "Base/Singleton.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/PreviewDescriptions.h"

class PreviewDescriptionConverterService
{
    MAKE_SINGLETON(PreviewDescriptionConverterService);

public:
    struct ConversionResult
    {
        PreviewDescription description;
        std::optional<float> visualFrontAngle;
    };
    ConversionResult
    convert(GenomeDescription const& genome, CollectionDescription&& phenotype, int startGeneIndex, std::optional<float> const& lastVisualFrontAngle) const;
};