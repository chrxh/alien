#pragma once

#include "Definitions.h"

class _SimulationScrollbars
{
public:
    void process(RealVector2D& worldCenter, RealRect const& worldRect, RealRect const& visibleWorldRect, RealRect const& viewRect);

private:
    enum class Orientation
    {
        Horizontal,
        Vertical
    };
    void processScrollbar(RealVector2D& worldCenter, RealRect const& worldRect, RealRect const& visibleWorldRect, RealRect const& viewRect, Orientation orientation);

    RealRect calcScrollbarRect(RealRect const& viewRect, Orientation orientation) const;

    void processEvents(RealVector2D& worldCenter, RealRect const& worldRect, RealRect const& viewRect, bool mouseCursorIntersectSliderbar, Orientation orientation);
    RealRect calcSliderbarRect(RealRect const& worldRect, RealRect const& visibleWorldRect, RealRect const& viewRect, Orientation orientation) const;
    bool doesMouseCursorIntersectSliderBar(RealRect const& scrollbarRect, RealRect const& sliderbarRect) const;

    std::optional<RealVector2D> _worldCenterForDragging;
    std::optional<Orientation> _orientationForDragging;
};