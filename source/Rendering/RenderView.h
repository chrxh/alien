#pragma once

#include <Base/Definitions.h>

// Part of the simulation that is rendered and how it is shown
struct RenderView
{
    IntVector2D viewSize;

    // Rendered pixels per world unit
    float zoomFactor = 1.0f;

    // Rendered pixels per screen pixel, greater than 1 if a picture is rendered with a higher resolution than the screen
    float renderScale = 1.0f;

    RealRect visibleWorldRect;
    bool cellDetailOverlay = false;

    // Screen pixels per world unit
    float getScreenZoomFactor() const { return zoomFactor / renderScale; }
};
