#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _SimulationScrollbar
{
public:
    enum class Orientation
    {
        Horizontal, Vertical
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

    std::string _id;
    Orientation _orientation = Orientation::Horizontal;
    SimulationController _simController;
    Viewport _viewport;
};