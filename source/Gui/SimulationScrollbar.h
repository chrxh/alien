#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _SimulationScrollbar
{
public:
    enum class Orientation
    {
        Horizontal,
        Vertical
    };
    _SimulationScrollbar(
        std::string const& id,
        Orientation orientation,
        SimulationController const& simController,
        Viewport const& viewport);

    void process(RealRect const& rect);

private:
    void processEvents(RealRect const& rect);
    RealRect calcSliderbarRect(RealRect const& scrollbarRect) const;
    bool doesMouseCursorIntersectSliderBar(RealRect const& rect) const;

    std::string _id;
    Orientation _orientation = Orientation::Horizontal;
    SimulationController _simController;
    Viewport _viewport;

    std::optional<RealVector2D> _worldCenterForDragging;
};