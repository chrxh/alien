#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"

class Scrollbar
{
public:
    enum class Orientation
    {
        Horizontal, Vertical
    };
    Scrollbar(Orientation orientation, SimulationController* simController);

    void draw(RealVector2D const& topLeft, RealVector2D const& bottomRight);
};