#pragma once

#include "Base/Singleton.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/PreviewDescriptions.h"

struct ConversionResult
{
    PreviewDescription description;
    std::optional<float> visualFrontAngle;
};

class PreviewDescriptionConverterService
{
    MAKE_SINGLETON(PreviewDescriptionConverterService);

public:
    ConversionResult
    convert(GenomeDescription const& genome, CollectionDescription&& phenotype, int startGeneIndex, std::optional<float> const& lastVisualFrontAngle) const;
};