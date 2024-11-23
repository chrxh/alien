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
    _SimulationScrollbar(std::string const& id, Orientation orientation, SimulationFacade const& simulationFacade);

    void process(RealRect const& rect);

private:
    void processEvents(RealRect const& rect);
    RealRect calcSliderbarRect(RealRect const& scrollbarRect) const;
    bool doesMouseCursorIntersectSliderBar(RealRect const& rect) const;

    std::string _id;
    Orientation _orientation = Orientation::Horizontal;
    SimulationFacade _simulationFacade;

    std::optional<RealVector2D> _worldCenterForDragging;
};