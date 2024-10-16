#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class Viewport
{
public:

    static void init(SimulationFacade const& simulationFacade);

    static float getZoomFactor();
    static void setZoomFactor(float zoomFactor);

    static RealVector2D getCenterInWorldPos();
    static void setCenterInWorldPos(RealVector2D const& worldCenter);

    static IntVector2D getViewSize();
    static void setViewSize(IntVector2D const& viewSize);

    static void zoom(IntVector2D const& viewPos, float factor);
    static float getZoomSensitivity();
    static void setZoomSensitivity(float value);

    static void centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos);
    static RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos);
    static RealVector2D mapWorldToViewPosition(RealVector2D worldPos, bool borderlessRendering = true);
    static RealRect getVisibleWorldRect();
    static bool isVisible(RealVector2D const& viewPos);

private:
    static SimulationFacade _simulationFacade;

    static float _zoomFactor;
    static float _zoomSensitivity;
    static RealVector2D _worldCenter;
    static IntVector2D _viewSize;
};
