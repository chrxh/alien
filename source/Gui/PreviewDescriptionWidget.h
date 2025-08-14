#pragma once

#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _PreviewDescriptionWidget
{
public:
    static PreviewDescriptionWidget create(PreviewDescriptionSettings const& settings);

    bool process(int tps, PreviewDescription const& desc, std::optional<int>& selectedGene, std::optional<int>& selectedNode);

private:
    _PreviewDescriptionWidget(PreviewDescriptionSettings const& settings);

    PreviewDescriptionSettings _settings;
    float _zoom = 20.0f;
};