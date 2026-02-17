#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/PreviewDesc.h>

struct ConversionResult
{
    PreviewDesc description;
    float frontAngle = 0;
    std::optional<float> visualFrontAngle;
};

class PreviewDescConverterService
{
    MAKE_SINGLETON(PreviewDescConverterService);

public:
    // Note: lastVisualFrontAngle currently deactivated
    ConversionResult convertToPreviewDesc(
        GenomeDesc const& genome,
        int startGeneIndex,
        Desc&& phenotype,
        std::optional<float> const& lastVisualFrontAngle) const;
};