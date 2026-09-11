#pragma once

#include <Base/Definitions.h>

#include "Definitions.h"

// Mapping between the world of a creature preview and the view it is drawn into
class PreviewViewport
{
public:
    float getZoom() const;
    void setZoom(float value);

    RealVector2D getWorldCenter() const;
    void setWorldCenter(RealVector2D const& value);

    RealVector2D getViewStartPos() const;
    void setViewStartPos(RealVector2D const& value);

    RealVector2D getViewSize() const;
    void setViewSize(RealVector2D const& value);

    RealVector2D mapWorldToViewPosition(RealVector2D const& worldPos) const;
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos) const;

    static float calcZoomToFitContent(RealVector2D const& contentSize, RealVector2D const& viewSize);

private:
    RealVector2D _worldCenter;
    float _zoom = 20.0f;
    RealVector2D _viewStartPos;
    RealVector2D _viewSize;
};
