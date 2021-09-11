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

    void processEvents();
    void draw(RealVector2D const& topLeft, RealVector2D const& size);
private:
    std::string _id;
    Orientation _orientation = Orientation::Horizontal;
    SimulationController _simController;
    Viewport _viewport;
};