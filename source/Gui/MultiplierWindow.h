#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"

enum class MultiplierMode
{
    Grid,
    Random
};

class _MultiplierWindow : public _AlienWindow
{
public:
    _MultiplierWindow(SimulationController const& simController, Viewport const& viewport);

private:
    void processIntern() override;

    SimulationController _simController;
    Viewport _viewport;

    MultiplierMode _mode = MultiplierMode::Grid;
};