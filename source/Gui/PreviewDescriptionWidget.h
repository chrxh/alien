#pragma once

#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _PreviewDescriptionWidget
{
public:
    static PreviewDescriptionWidget create(PreviewDescriptionSettings const& settings, GenomeTabEditData const& editData);

    bool process(int tps, PreviewDescription const& desc);

private:
    _PreviewDescriptionWidget(PreviewDescriptionSettings const& settings, GenomeTabEditData const& editData);

    PreviewDescriptionSettings _settings;
    GenomeTabEditData _editData;
    float _zoom = 20.0f;
};