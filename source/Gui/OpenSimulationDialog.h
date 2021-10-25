#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _OpenSimulationDialog
{
public:
    _OpenSimulationDialog(
        SimulationController const& simController,
        StatisticsWindow const& statisticsWindow,
        Viewport const& viewport);

    void process();

    void show();

private:
    SimulationController _simController;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;
};

