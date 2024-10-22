#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "Base/Singleton.h"

class Viewport
{
    MAKE_SINGLETON(Viewport);

public:
    void setup(SimulationFacade const& simulationFacade);

    float getZoomFactor();
    void setZoomFactor(float zoomFactor);

    RealVector2D getCenterInWorldPos();
    void setCenterInWorldPos(RealVector2D const& worldCenter);

    IntVector2D getViewSize();
    void setViewSize(IntVector2D const& viewSize);

    void zoom(IntVector2D const& viewPos, float factor);
    float getZoomSensitivity();
    void setZoomSensitivity(float value);

    void centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos);
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos);
    RealVector2D mapWorldToViewPosition(RealVector2D worldPos, bool borderlessRendering = true);
    RealRect getVisibleWorldRect();
    bool isVisible(RealVector2D const& viewPos);

private:
    SimulationFacade _simulationFacade;

    float _zoomFactor = 1.0f;
    float _zoomSensitivity = 1.07f;
    RealVector2D _worldCenter;
    IntVector2D _viewSize;
};
