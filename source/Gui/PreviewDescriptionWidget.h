#pragma once

#include "EngineInterface/PreviewDescriptions.h"

#include "Definitions.h"

class _PreviewDescriptionWidget
{
public:
    static PreviewDescriptionWidget create(GenomeTabEditData const& editData);

    bool process(PreviewDescription const& desc);

private:
    _PreviewDescriptionWidget(GenomeTabEditData const& editData);

    GenomeTabEditData _editData;
    float _zoom = 20.0f;
    std::optional<RealVector2D> _lastWindowSize;
    bool _initialScrollPositionSet = false;
};