#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>

#include "Definitions.h"

class Viewport
{
    MAKE_SINGLETON(Viewport);

public:
    void setup();

    // World units to rendered pixels
    float getZoomFactor();
    void setZoomFactor(float zoomFactor);

    // Rendered pixels per screen pixel. Greater than 1 while a picture is rendered with a higher resolution
    // than the view. Everything with a size in pixels needs to be scaled by this factor to keep its appearance.
    float getRenderScale();
    void setRenderScale(float value);

    // World units to screen pixels, i.e. the zoom factor an equivalent view on screen would have.
    // Should be used for decisions that depend on the visual zoom level instead of the rendered resolution.
    float getScreenZoomFactor();

    RealVector2D getCenterInWorldPos();
    void setCenterInWorldPos(RealVector2D const& worldCenter);

    IntVector2D getViewSize();
    void setViewSize(IntVector2D const& viewSize);

    void zoom(IntVector2D const& viewPos, float factor);
    float getZoomSensitivity();
    void setZoomSensitivity(float value);

    void moveCenter(RealVector2D const& startWorldPosition, IntVector2D const& endViewPos);
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos);
    RealVector2D mapWorldToViewPosition(RealVector2D worldPos, bool borderlessRendering = true);
    RealRect getVisibleWorldRect();
    bool isVisible(RealVector2D const& viewPos);

private:
    float _zoomFactor = 1.0f;
    float _renderScale = 1.0f;
    float _zoomSensitivity = 1.04f;
    RealVector2D _centerInWorldPos;
    IntVector2D _viewSize;
};
