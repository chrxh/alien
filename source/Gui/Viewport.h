#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"

class _Viewport
{
public:
    float getZoomFactor() const;
    void setZoomFactor(float zoomFactor);

    RealVector2D getCenterInWorldPos() const;
    void setCenterInWorldPos(RealVector2D const& worldCenter);

    IntVector2D getViewSize() const;
    void setViewSize(IntVector2D const& viewSize);

    void zoom(IntVector2D const& viewPos, float factor);
    float getZoomSensitivity() const;
    void setZoomSensitivity(float value);

    void centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos);
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos) const;
    RealVector2D mapWorldToViewPosition(RealVector2D const& worldPos) const;
    RealRect getVisibleWorldRect() const;

private:
    float _zoomFactor = 1.0f;
    float _zoomSensitivity = 1.03f;
    RealVector2D _worldCenter;
    IntVector2D _viewSize;
};