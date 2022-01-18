#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _MultiplierWindow : public _AlienWindow
{
public:
    _MultiplierWindow(SimulationController const& simController, Viewport const& viewport);

private:
    void processIntern() override;

    SimulationController _simController;
    Viewport _viewport;
};