#pragma once

#include <Base/Singleton.h>

#include <optional>

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
    ConversionResult
    convertToPreviewDesc(GenomeDesc const& genome, int startGeneIndex, Desc&& phenotype, std::optional<float> const& lastVisualFrontAngle = std::nullopt) const;
};
